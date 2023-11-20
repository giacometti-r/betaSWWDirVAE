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
        self.dense1 = Dense(400, activation='relu', kernel_initializer=initializer)
        self.dense2 = Dense(500, activation='relu', kernel_initializer=initializer)
        self.dense_mu = Dense(latent_dim, activation='relu', kernel_initializer=initializer)
        self.dense_logvar = Dense(latent_dim, activation='relu', kernel_initializer=initializer)

    def sample(self, mu, logvar):
        std = tf.math.exp(0.5*logvar)
        epsilon = tf.random.uniform(shape=std.shape)
        return mu + epsilon*std

    def call(self, x):
        x = self.dense1(x)
        #x = self.dense2(x)
        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        z = self.sample(mu, logvar)
        return z, mu, logvar
    

class Decoder(Model):
    def __init__(self, original_shape):
        super(Decoder, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.dense1 = Dense(400, activation='relu',kernel_initializer=initializer)
        self.dense2 = Dense(original_shape, activation='sigmoid',kernel_initializer=initializer)

    def call(self, x):
        x_hat = self.dense1(x)
        x_hat = self.dense2(x_hat)
        return x_hat
    
class GVAE(Model):
    def __init__(self, latent_dim, original_dim):
        super(GVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(original_dim)

    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss

    def call(self, inputs):
        z, mu, logvar = self.encoder(inputs)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
@tf.function
def ELBO(log_likelihood_loss, y_pred, y_true, mu, logvar):
    ll_loss = log_likelihood_loss(y_true, y_pred)
    
    kld_loss = -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.pow(mu,2) - tf.math.exp(logvar))

    return (tf.math.add(ll_loss, tf.math.maximum(0.0,kld_loss))), ll_loss, kld_loss

def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_reshaped = x_train.reshape((-1,28*28)).astype("float32") / 255.0
x_test_reshaped = x_test.reshape((-1,28*28)).astype("float32") / 255.0

print('GPU:', tf.config.list_physical_devices('GPU'))
tf.config.run_functions_eagerly(False)

latent_dim = 50
original_dim = 28*28

model = GVAE(latent_dim, original_dim)

log_likelihood_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum', from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

batch_size = 100

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
            prediction, mu, logvar = model(x_batch_train, training=True)
            loss_value, ll_loss, kld_loss = ELBO(log_likelihood_loss, prediction, x_batch_train, mu, logvar)

        grads = tape.gradient(loss_value, model.trainable_weights)
        grads = [ClipIfNotNone(grad) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 250 == 0:
            print(f"Training loss at step {step}: {float(loss_value):.4f}")
            print(f"LL loss at step {step}: {float(ll_loss):.4f}")
            print(f"kld loss at step {step}: {float(kld_loss):.4f}\n")


    val_loss = []
    for step, (x_batch_test) in enumerate(test_dataset):
        val_prediction, mu, logvar = model(x_batch_test, training=False)
        val_loss_value, val_ll_loss, val_kld_loss = ELBO(log_likelihood_loss, val_prediction, x_batch_test, mu, logvar)
        val_loss.append(val_loss_value.numpy())
    val_loss = np.mean(np.array(val_loss))
    print('AVERAGE VALIDATION LOSS:', val_loss)
    print()

    train_loss_list.append(loss_value.numpy())
    kld_loss_list.append(kld_loss.numpy())
    ll_loss_list.append(ll_loss.numpy())

    image, img_mu, img_logvar = model(tf.convert_to_tensor(x_train[0].reshape((1,28*28))/255.0), training=False)
    if epoch == 0:
        plt.imshow(x_train[0])
        plt.show()
        plt.cla()
        #plt.savefig('/reconstructed_images/original.png')
    
    plt.imshow(image[0].numpy().reshape((28,28)))
    plt.show()
    plt.cla()
    #plt.savefig(f'/reconstructed_images/epoch_{epoch}.png')

    if val_loss_value.numpy() < lowest_loss:
        model.save_weights('/weights/my_model', overwrite=True, save_format='tf', options=None)
        lowest_loss = val_loss_value.numpy()

df_loss = pd.DataFrame(np.vstack([train_loss_list, kld_loss_list, ll_loss_list]).T, columns=['total_loss', 'kld_loss', 'll_loss'])
df_loss.to_csv('loss.csv', index=False)