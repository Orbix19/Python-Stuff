import os
import numpy as np
import re
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class GoBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def apply_move(self, player, move):
        stone = 1 if player == 'B' else -1
        x, y = self.sgf_to_coords(move)
        if 0 <= x < self.size and 0 <= y < self.size:
            self.board[x, y] = stone
            self.check_captures(x, y, stone)

    def sgf_to_coords(self, sgf_coords):
        x = ord(sgf_coords[0]) - ord('a')
        y = ord(sgf_coords[1]) - ord('a')
        return x, y

    def check_captures(self, x, y, stone):
        pass

    def get_board_layers(self):
        black_stones = (self.board == 1).astype(int)
        white_stones = (self.board == -1).astype(int)
        empty_positions = (self.board == 0).astype(int)
        return np.stack([black_stones, white_stones, empty_positions], axis=-1)

def parse_sgf(sgf_content):
    pattern = re.compile(r';(B|W)\[(\w{2})\]')
    moves = pattern.findall(sgf_content)
    return moves

def process_sgf_file(file_path):
    with open(file_path, 'r') as file:
        sgf_content = file.read()
    moves = parse_sgf(sgf_content)
    board = GoBoard()
    board_states = []
    labels = []

    for i in range(len(moves) - 1):
        player, move = moves[i]
        board.apply_move(player, move)
        board_state = board.get_board_layers()
        next_move = moves[i + 1][1]
        label = ord(next_move[0]) - ord('a') + 19 * (ord(next_move[1]) - ord('a'))
        board_states.append(board_state)
        labels.append(label)

    return np.array(board_states), np.array(labels)

class SGFDataGenerator(Sequence):
    def __init__(self, file_paths, batch_size=8, dim=(19, 19, 3), shuffle=True, return_context=False):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.return_context = return_context
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_file_paths = [self.file_paths[k] for k in indices]
        X, y, file_names = self.__data_generation(batch_file_paths)
        if self.return_context:
            return X, y, file_names
        else:
            return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_file_paths):
        X = []
        y = []
        file_names = []

        for file_path in batch_file_paths:
            board_states, labels = process_sgf_file(file_path)
            X.extend(board_states)
            y.extend(labels)
            file_names.extend([file_path] * len(board_states))

        X = np.array(X)
        y = np.array(y)
        y = to_categorical(y, num_classes=361)

        return X, y, file_names

def build_overfit_go_cnn(input_shape=(19, 19, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    
    outputs = Dense(361, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_regularized_go_cnn(input_shape=(19, 19, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(361, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Prepare data
folder_path = '/remote_home/Python-Stuff/GoNN/Go4Go'
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.sgf')]

# Split data into training and testing sets
train_files, test_files = train_test_split(file_paths, test_size=0.2)
print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")

train_generator = SGFDataGenerator(train_files, batch_size=8)
test_generator = SGFDataGenerator(test_files, batch_size=8)

# Build and compile the overfitting model
model = build_overfit_go_cnn()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# Train the overfitting model
history = model.fit(train_generator, validation_data=test_generator, epochs=50, callbacks=[model_checkpoint, reduce_lr], verbose=1)

# # Build and compile the regularized model
# model = build_regularized_go_cnn()
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the regularized model
# history = model.fit(train_generator, validation_data=test_generator, epochs=50, callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=1)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_over_epochs.png')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_over_epochs.png')

    plt.show()

    # Analyzing the learning rate
    lr_history = history.history.get('lr', [])
    if lr_history:
        plt.figure(figsize=(6, 4))
        plt.plot(lr_history, label='Learning Rate')
        plt.title('Learning Rate Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.savefig('learning_rate_over_epochs.png')

        plt.show()

plot_history(history)

# Evaluate the model on the test set with batching
def evaluate_model_in_batches(model, data_generator, batch_size=8):
    y_true = []
    y_pred = []
    for i in range(len(data_generator)):
        X_batch, y_batch = data_generator[i]
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_batch = model.predict(X_batch)
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = evaluate_model_in_batches(model, test_generator)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Display metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display metrics in a table
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(metrics)

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title('Evaluation Metrics')
plt.savefig('evaluation_metrics.png')
plt.show()
