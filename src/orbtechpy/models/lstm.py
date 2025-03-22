from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import time
import tensorflow as tf
import matplotlib.pyplot as plt

class LSTMModel(Model):
    def __init__(self, units, num_classes, input_shape, **kwargs):
        super(LSTMModel, self).__init__(**kwargs)
        self.units = units
        self.num_classes = num_classes
        self.input_shape = input_shape
        # Define layers
        self.normalization = layers.Normalization(axis=-1)

        self.conv1d = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(pool_size=2)

        self.lstm1 = layers.LSTM(self.units, return_sequences=True)

        self.dropout = layers.Dropout(0.1)  # Dropout with a 10% dropout rate
        self.dense1 = layers.Dense(64, activation='tanh')
        self.dense2 = layers.Dense(self.num_classes, activation='linear')

        # Define the loss functions
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.kl_loss = tf.keras.losses.KLDivergence(reduction='sum_over_batch_size', name='kl_divergence')

    def build(self, input_shape):
        # Build layers properly with explicit input shapes
        super(LSTMModel, self).build(input_shape)
        self.normalization.build(input_shape)
        
        # self.conv1d.build(input_shape)
        # self.pool1.build(self.conv1d.compute_output_shape(input_shape))
        # self.lstm1.build(self.pool1.compute_output_shape(self.conv1d.compute_output_shape(input_shape)))
        # self.dense.build(self.lstm1.compute_output_shape(self.pool1.compute_output_shape(self.conv1d.compute_output_shape(input_shape))))

        self.lstm1.build(input_shape)
        self.dropout.build(self.lstm1.compute_output_shape(input_shape))
        self.dense1.build(self.lstm1.compute_output_shape(input_shape))
        self.dense2.build(self.dense1.compute_output_shape(input_shape))

    def adapt(self, data):
        """Adapt the normalization layer using training data."""
        self.normalization.adapt(data)

    def call(self, inputs, training=True):
        x = self.normalization(inputs)
        # x = self.conv1d(x)
        # x = self.pool1(x)
        x = self.lstm1(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)  # Ensure dropout is used
        return self.dense2(x)

    def compile(self, use_dynamics_loss=False, lam=1e-3):
        """Compile the model with MSE loss."""
        optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

        if use_dynamics_loss:
            def combined_loss(y_true, y_pred):
                return self.dynamics_loss(y_true, y_pred, lam)
            super(LSTMModel, self).compile(optimizer=optimizer, loss=combined_loss, metrics=["mae"])
        else:
            super(LSTMModel, self).compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    
    def dynamics_loss(self,y_true, y_pred,lam=1e-3):
        # data_loss = self.mse_loss(y_true, y_pred)
        data_loss = self.mse_loss(y_true, y_pred)
        # accel_loss = acceleration_deviation(y_pred)
        total_loss = data_loss #+ accel_loss * lam
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# def acceleration_deviation(y_pred,perturbation=1e-6):
#     mu = 398600.4418
#     rx = y_pred[:,:, 0]
#     ry = y_pred[:,:, 1]
#     rz = y_pred[:,:, 2]

#     vx = y_pred[:,:, 3]
#     vy = y_pred[:,:, 4]
#     vz = y_pred[:,:, 5]

#     radius = tf.norm([rx, ry, rz], axis=0)

#     # Calculate the magnitude of the acceleration
#     ax = -mu * rx / radius ** 3
#     ay = -mu * ry / radius ** 3 
#     az = -mu * rz / radius ** 3

#     accel1 = tf.stack([ax, ay, az], axis=1)
    
    
#     return tf.reduce_mean(deviation)

class LiveLossPlot(tf.keras.callbacks.Callback):
   
    def __init__(self, interval=10, save_path="loss_change_plot.png", zoom_threshold=1e-3, zoom_factor=4, min_ylim=-1, max_ylim=1,window = 10):
        super().__init__()
        self.interval = interval  # Save interval
        self.save_path = save_path
        self.zoom_threshold = zoom_threshold  # Threshold for zooming
        self.zoom_factor = zoom_factor  # Zoom factor for y-axis adjustment
        self.history = {"loss": [], "val_loss": []}
        self.loss_change = {"loss": [], "val_loss": []}
        self.ylim_min = min_ylim  # Initial minimum of the y-axis (you can adjust this)
        self.ylim_max = max_ylim  # Initial maximum of the y-axis (you can adjust this)
        self.window = window
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        curr_loss = logs.get("loss")
        curr_val_loss = logs.get("val_loss")

        # Store current loss
        self.history["loss"].append(curr_loss)
        self.history["val_loss"].append(curr_val_loss)

        self.epochs.append(epoch)

        # Compute change (first epoch has no change value)
        if len(self.history["loss"]) > 1:
            loss_change = self.history["loss"][-1] - self.history["loss"][-2]
            val_loss_change = self.history["val_loss"][-1] - self.history["val_loss"][-2]

            self.loss_change["loss"].append(loss_change)
            self.loss_change["val_loss"].append(val_loss_change)

        if (epoch + 1) % self.interval == 0 and len(self.loss_change["loss"]) > 1:
            plt.figure(figsize=(8, 5))
            if epoch > self.window:
                plt.plot(self.epochs[-self.window:],self.loss_change["loss"][-self.window:], label="Change in Train Loss", color='blue')
                plt.plot(self.epochs[-self.window:],self.loss_change["val_loss"][-self.window:], label="Change in Validation Loss", color='orange')
            else:
                plt.plot(self.epochs[1:],self.loss_change["loss"], label="Change in Train Loss", color='blue')
                plt.plot(self.epochs[1:],self.loss_change["val_loss"], label="Change in Validation Loss", color='orange')

            plt.xlabel("Epoch")
            plt.ylabel("Loss Change")
            plt.legend()
            plt.title("Change in Training and Validation Loss")
            plt.axhline(0, color='gray', linestyle='--')  # Reference line at 0
            plt.grid()

            # Continuously zoom in if the loss change is sufficiently small
            if abs(self.loss_change["loss"][-1]) < self.zoom_threshold and abs(self.loss_change["val_loss"][-1]) < self.zoom_threshold:
                print("\nZooming in...")
                # Ensure the y-limits don't go too extreme
                self.ylim_min = max(self.ylim_min / self.zoom_factor, -1)
                self.ylim_max = min(self.ylim_max / self.zoom_factor, 1)

                # plt.ylim(self.ylim_min, self.ylim_max)

            # Save the updated figure (overwrite previous file)
            plt.savefig(self.save_path)
            plt.close()
            print(f"\nUpdated loss change plot saved at epoch {epoch+1}")

    def on_train_begin(self, logs=None):
        # Record start time when training begins
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        # Save the final figure at the end of training

        self.end_time = time.time()

        # Calculate total training time
        training_time = self.end_time - self.start_time
        training_minutes = training_time / 60  # Convert to minutes
        training_seconds = training_time % 60  # Remaining seconds

        print(f"Training completed in {training_minutes:.2f} minutes and {training_seconds:.2f} seconds.")
        
        plt.figure(figsize=(8, 5))
        plt.plot(self.epochs[-self.window:],self.loss_change["loss"][-self.window:], label="Change in Train Loss", color='blue')
        plt.plot(self.epochs[-self.window:],self.loss_change["val_loss"][-self.window:], label="Change in Validation Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss Change")
        plt.legend()
        # plt.ylim(self.ylim_min, self.ylim_max)
        plt.title("Change in Training and Validation Loss")
        plt.axhline(0, color='gray', linestyle='--')
        plt.grid()
        plt.savefig(f"_final{self.window}" + self.save_path)
        print("Training completed. Loss change plot saved.")

        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_change["loss"], label="Change in Train Loss", color='blue')
        plt.plot(self.loss_change["val_loss"], label="Change in Validation Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss Change")
        plt.legend()
        # plt.ylim(self.ylim_min, self.ylim_max)
        plt.title("Change in Training and Validation Loss")
        plt.axhline(0, color='gray', linestyle='--')
        plt.grid()
        plt.savefig( self.save_path)
        print("Training completed. Loss change plot saved.")
