import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Cargar datos Fashion MNIST
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

# Normalización de datos
X_train_full = X_train_full / 255.0
X_test_full = X_test_full / 255.0

# Dividir en entrenamiento y validación
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Convertir a tipo float32 de JAX
X_train = jnp.array(X_train, dtype=jnp.float32)
X_valid = jnp.array(X_valid, dtype=jnp.float32)
X_test = jnp.array(X_test_full, dtype=jnp.float32)
y_train = jnp.array(y_train, dtype=jnp.int32)
y_valid = jnp.array(y_valid, dtype=jnp.int32)
y_test = jnp.array(y_test_full, dtype=jnp.int32)

# Definir nombres de las clases
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Definir el modelo en Flax
class FashionMNISTModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Aplanar las imágenes 28x28
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)  # 10 clases
        return x

# Inicializar modelo
model = FashionMNISTModel()
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones([1, 28, 28]))  # Inicializar parámetros

# Definir función de pérdida
def loss_fn(params, x, y):
    logits = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean()

# Definir optimizador
optimizer = optax.sgd(learning_rate=0.1)
opt_state = optimizer.init(params)

# Paso de entrenamiento
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Entrenamiento del modelo
num_epochs = 10
for epoch in range(num_epochs):
    params, opt_state, loss = train_step(params, opt_state, X_train, y_train)
    print(f"Época {epoch+1}, Pérdida: {loss:.4f}")

# Predicción en los primeros 3 elementos de test
def predict(params, x):
    logits = model.apply(params, x)
    return jnp.argmax(logits, axis=1)

y_pred = predict(params, X_test[:3])
print("Predicciones:", np.array(class_names)[y_pred])

# Mostrar imágenes y predicciones
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.axis("off")
    plt.imshow(X_test_full[i], cmap="gray")
    plt.title(class_names[y_pred[i]])

plt.show()
