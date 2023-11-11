import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
tfd = tfp.distributions
# run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
tf.random.set_seed(42)


class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=42)
        self.dense1 = Dense(500, activation='relu', kernel_initializer=initializer)
        self.dense2 = Dense(500, activation='relu', kernel_initializer=initializer)
        self.dense3 = Dense(128, activation='relu', kernel_initializer=initializer)
        self.dense4 = Dense(latent_dim, activation='softplus', kernel_initializer=initializer)

    def sample(self, alpha_hat, alpha, beta):
        shape = (alpha_hat.get_shape().as_list()[-2],alpha_hat.get_shape().as_list()[-1])
        u = tf.random.uniform(shape=shape, minval=0, maxval=1)
        v = tf.math.multiply(u,alpha)
        v = tf.math.multiply(v, tf.math.exp(tf.math.lgamma(alpha)))
        v = tf.math.pow(v, tf.math.divide(1.0,alpha))
        v = tf.math.divide(v,beta)
        z ,_= tf.linalg.normalize(v)
        
        return z
    
    def call(self, inputs):
        x = inputs[0]
        alpha = inputs[1]
        beta = inputs[2]
        #x = tf.reshape(x, (-1,28*28))
        alpha_hat = self.dense1(x)
        alpha_hat = self.dense2(alpha_hat)
        #alpha_hat = self.dense3(alpha_hat)
        alpha_hat = self.dense4(alpha_hat)
        z = self.sample(alpha_hat, alpha, beta)
        return z, alpha_hat

class Decoder(Model):
    def __init__(self, original_shape):
        super(Decoder, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=42)
        self.dense1 = Dense(500, activation='relu',kernel_initializer=initializer)
        self.dense2 = Dense(256, activation='relu',kernel_initializer=initializer)
        self.dense3 = Dense(512, activation='relu',kernel_initializer=initializer)
        self.dense4 = Dense(original_shape, activation='sigmoid',kernel_initializer=initializer)

    def call(self, x):
        x_hat = self.dense1(x)
        #x_hat = self.dense2(x_hat)
        #x_hat = self.dense3(x_hat)
        x_hat = self.dense4(x_hat)
        #x_hat = tf.reshape(x_hat,[-1,28,28])
        return x_hat
    
class DirVAE(Model):
    def __init__(self, latent_dim, original_dim):
        super(DirVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(original_dim)
    
    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss

    def call(self, inputs):
        z, alpha_hat = self.encoder(inputs)
        x_hat = self.decoder(z)
        return x_hat, alpha_hat

@tf.function
def ELBO(log_likelihood_loss, y_pred, y_true, alpha, alpha_hat):
    ll_loss = log_likelihood_loss(y_true, y_pred)

    kld_loss = tf.math.subtract(tf.math.lgamma(alpha), tf.math.lgamma(alpha_hat))
    kld_loss = tf.math.add(kld_loss, tf.math.multiply(tf.math.subtract(alpha_hat, alpha),tf.math.digamma(alpha_hat)))
    kld_loss = tf.reduce_sum(kld_loss)
    
    # kld_loss = tf.reduce_sum( 
    #     tf.math.lgamma(alpha) -
    #     tf.math.lgamma(alpha_hat) +
    #     (alpha_hat - alpha) * tf.math.digamma(alpha_hat)
    # )
    
    return (tf.math.add(ll_loss, tf.math.maximum(0.0,kld_loss))), ll_loss, kld_loss

@tf.function
def update_alpha_mme(alpha, samples=50):
    
    dirichlet = tfd.Dirichlet(alpha)
    p_set = dirichlet.sample([samples])
    N, K = p_set.shape

    mu1_tilde = tf.math.reduce_mean(p_set, axis=0)
    mu2_tilde = tf.math.reduce_mean(tf.math.pow(p_set,2), axis=0)

    S = tf.math.reduce_mean(tf.math.divide((tf.math.subtract(mu1_tilde, mu2_tilde)), (tf.math.subtract(mu2_tilde, tf.math.pow(mu1_tilde,2)))), axis=0)

    alpha = tf.math.multiply(tf.math.divide(S,N), tf.math.reduce_sum(p_set, axis=0))

    return alpha

def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1000, 1000)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_reshaped = x_train.reshape((-1,28*28)).astype("float32") / 255
x_test_reshaped = x_test.reshape((-1,28*28)).astype("float32") / 255

print('GPU:', tf.config.list_physical_devices('GPU'))
tf.config.run_functions_eagerly(False)

latent_dim = 50
original_dim = 28*28

model = DirVAE(latent_dim, original_dim)

log_likelihood_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum')
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

batch_size = 100
alpha = (1.0-(1.0/latent_dim)) * tf.ones((latent_dim,))
beta = 1.0 * tf.ones((latent_dim,))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_reshaped))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_reshaped))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

lowest_loss = np.inf
epochs = 1500
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            prediction, alpha_hat = model([x_batch_train, alpha, beta], training=True)
            loss_value, ll_loss, kld_loss = ELBO(log_likelihood_loss, prediction, x_batch_train, alpha, alpha_hat)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)
        
        grads = [ClipIfNotNone(grad) for grad in grads]


        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Log every 200 batches.
        if step % 250 == 0:
            print(
                "Training loss at step %d: %.4f"
                % (step, float(loss_value))
            )
            print(
                "LL loss at step %d: %.4f"
                % (step, float(ll_loss))
            )
            print(
                "kld loss at step %d: %.4f"
                % (step, float(kld_loss))
            )
            # print(
            #     "Validation loss at step %d: %.4f"
            #     % (step, float(val_loss_value))
            # )
            print()

    val_loss = []
    for step, (x_batch_test) in enumerate(test_dataset):
        val_prediction, val_alpha_hat = model([x_batch_test, alpha, beta], training=False)
        val_loss_value, val_ll_loss, val_kld_loss = ELBO(log_likelihood_loss, val_prediction, x_batch_test, alpha, val_alpha_hat)
        val_loss.append(val_loss_value.numpy())
    val_loss = np.mean(np.array(val_loss))
    print('AVERAGE VALIDATION LOSS:', val_loss)
    print()
    
    ##UPDATE ALPHA    
    if epoch % 10 == 0 and epoch != 0:
        alpha = update_alpha_mme(alpha)
        
    if loss_value.numpy() < lowest_loss:
        model.save_weights('my_model', overwrite=True, save_format='tf', options=None)
        lowest_loss = loss_value.numpy()
    print('Alpha:', alpha.numpy())

print('FINAL ALPHA:', alpha.numpy())
FINAL_ALPHA = alpha