import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import Loss
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

tfd = tfp.distributions
tf.random.set_seed(42)

class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.dense1 = Dense(500, activation='relu', kernel_initializer=initializer)
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(500, activation='relu', kernel_initializer=initializer)
        self.dropout2 = Dropout(0.5)
        self.dense3 = Dense(128, activation='relu', kernel_initializer=initializer)
        self.dropout3 = Dropout(0.5)
        self.dense4 = Dense(latent_dim, activation='softplus', kernel_initializer=initializer)

    def sample(self, alpha_hat, alpha, beta):
        shape = (alpha_hat.get_shape().as_list()[-2],alpha_hat.get_shape().as_list()[-1])
        u = tf.random.uniform(shape=shape, minval=0, maxval=1)
        v = tf.math.multiply(u,alpha_hat)
        v = tf.math.multiply(v, tf.math.exp(tf.math.lgamma(alpha_hat)))
        v = tf.math.pow(v, tf.math.divide(1.0,alpha_hat))
        v = tf.math.divide(v,beta)
        z = tf.math.divide(v,tf.math.reduce_sum(v)) #sum to one

        return z, v

    def call(self, inputs):
        x = inputs[0]
        alpha = inputs[1]
        beta = inputs[2]
        alpha_hat = self.dense1(x)
        #alpha_hat = self.dropout1(alpha_hat)
        alpha_hat = self.dense2(alpha_hat)
        #alpha_hat = self.dropout2(alpha_hat)
        #alpha_hat = self.dense3(alpha_hat)
        #alpha_hat = self.dropout3(alpha_hat)
        alpha_hat = self.dense4(alpha_hat)
        z, v = self.sample(alpha_hat, alpha, beta)
        return z, alpha_hat, v
    
class Decoder(Model):
    def __init__(self, original_shape):
        super(Decoder, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.dense1 = Dense(500, activation='relu',kernel_initializer=initializer)
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(256, activation='relu',kernel_initializer=initializer)
        self.dropout2 = Dropout(0.5)
        self.dense3 = Dense(512, activation='relu',kernel_initializer=initializer)
        self.dropout3 = Dropout(0.5)
        self.dense4 = Dense(original_shape, activation='sigmoid',kernel_initializer=initializer)

    def call(self, x):
        x_hat = self.dense1(x)
        #x_hat = self.dropout1(x_hat)
        #x_hat = self.dense2(x_hat)
        #x_hat = self.dropout2(x_hat)
        #x_hat = self.dense3(x_hat)
        #x_hat = self.dropout3(x_hat)
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
        z, alpha_hat, v = self.encoder(inputs)
        x_hat = self.decoder(z)
        return x_hat, z, alpha_hat, v
    
@tf.function
def ELBO(log_likelihood_loss, y_pred, y_true, alpha, alpha_hat):
    ll_loss = log_likelihood_loss(y_true, y_pred)
    
    beta = 1.0
    kld_loss = tf.math.subtract(tf.math.lgamma(alpha), tf.math.lgamma(alpha_hat))
    kld_loss = tf.math.add(kld_loss, tf.math.multiply(tf.math.subtract(alpha_hat, alpha),tf.math.digamma(alpha_hat)))
    kld_loss = tf.reduce_sum(kld_loss)

    return (tf.math.add(ll_loss, tf.math.multiply(beta,tf.math.maximum(0.0,kld_loss)))), ll_loss, kld_loss

@tf.function
def update_alpha_mme(z, samples=1, epsilon=1e-13):
    
    epsilon = tf.convert_to_tensor(epsilon)
    dirichlet = tfd.Dirichlet(z)
    p_set = dirichlet.sample([samples])
    _, N, K = p_set.shape
    p_set = tf.reshape(p_set, (100,50,))

    mu1_tilde = tf.math.reduce_mean(p_set, axis=0)
    mu2_tilde = tf.math.reduce_mean(tf.math.pow(p_set,2), axis=0)

    S = tf.math.reduce_mean(tf.math.divide((tf.math.subtract(mu1_tilde, mu2_tilde)), (tf.math.subtract(mu2_tilde, tf.math.pow(mu1_tilde,2))) + epsilon), axis=0)

    alpha = tf.math.multiply(tf.math.divide(S,N), tf.math.reduce_sum(p_set, axis=0)) + epsilon

    return alpha

def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -10, 10)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_reshaped = x_train.reshape((-1,28*28)).astype("float32") / 255
x_test_reshaped = x_test.reshape((-1,28*28)).astype("float32") / 255

print('GPU:', tf.config.list_physical_devices('GPU'))
tf.config.run_functions_eagerly(False)

latent_dim = 50
original_dim = 28*28

model = DirVAE(latent_dim, original_dim)

log_likelihood_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum', from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

batch_size = 100
alpha = (1.0-(1.0/latent_dim)) * tf.ones((latent_dim,))
beta = 1.0 * tf.ones((latent_dim,))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_reshaped))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_reshaped))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

lowest_loss = np.inf
epochs = 300
count = 0
train_loss_list = []
kld_loss_list = []
ll_loss_list = []

for epoch in range(epochs):
    print('___________________________')
    print(f'_____EPOCH_{epoch}________')
    print('___________________________')
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            prediction, z, alpha_hat, v = model([x_batch_train, alpha, beta], training=True)
            loss_value, ll_loss, kld_loss = ELBO(log_likelihood_loss, prediction, x_batch_train, alpha, alpha_hat)

        grads = tape.gradient(loss_value, model.trainable_weights)
        grads = [ClipIfNotNone(grad) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 250 == 0:
            print(f"Training loss at step {step}: {float(loss_value):.4f}")
            print(f"LL loss at step {step}: {float(ll_loss):.4f}")
            print(f"kld loss at step {step}: {float(kld_loss):.4f}\n")


    val_loss = []
    for step, (x_batch_test) in enumerate(test_dataset):
        val_prediction, val_z, val_alpha_hat, val_v = model([x_batch_test, alpha, beta], training=False)
        val_loss_value, val_ll_loss, val_kld_loss = ELBO(log_likelihood_loss, val_prediction, x_batch_test, alpha, val_alpha_hat)
        val_loss.append(val_loss_value.numpy())
    val_loss = np.mean(np.array(val_loss))
    print('AVERAGE VALIDATION LOSS:', val_loss)
    print()

    ##UPDATE ALPHA
    # if epoch % 20 == 0 and epoch != 0 and count <= 2:
    #     alpha = update_alpha_mme(z)
    #     print('Alpha:', alpha.numpy())
    #     count += 1

    train_loss_list.append(loss_value.numpy())
    kld_loss_list.append(kld_loss.numpy())
    ll_loss_list.append(ll_loss.numpy())
    
    inputs = [
        tf.convert_to_tensor(x_train[0].reshape((1,28*28)) / 255.0),
        alpha,
        beta
    ]

    image, img_z, img_alpha_hat, img_v = model(inputs)
    if epoch == 0:
        plt.imshow(x_train[0] / 255.0)
        plt.savefig('/reconstructed_images/original.png')
    
    plt.imshow(image[0].numpy().reshape((28,28)))
    plt.title(f'epoch_{epoch}')
    plt.savefig(f'/reconstructed_images/epoch_{epoch}.png')

    if val_loss_value.numpy() < lowest_loss:
        model.save_weights('/weights/my_model', overwrite=True, save_format='tf', options=None)
        lowest_loss = val_loss_value.numpy()


df_loss = pd.DataFrame(np.vstack([train_loss_list, kld_loss_list, ll_loss_list]).T, columns=['total_loss', 'kld_loss', 'll_loss'])
df_loss.to_csv('loss.csv', index=False)
print('FINAL ALPHA:', alpha.numpy())
FINAL_ALPHA = alpha

