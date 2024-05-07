import os
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

class GoBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def apply_move(self, player, move):
        stone = 1 if player == 'B' else -1
        x, y = self.sgf_to_coords(move)
        self.board[x, y] = stone
        self.check_captures(x, y, stone)

    def sgf_to_coords(self, sgf_coords):
        x = ord(sgf_coords[0]) - ord('a')
        y = ord(sgf_coords[1]) - ord('a')
        return x, y

    def check_captures(self, x, y, stone):
        # Capture checking and group removal not shown for brevity
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

def read_sgf_files(folder_path):
    all_moves = []
    all_states = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".sgf"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                sgf_content = file.read()
            moves = parse_sgf(sgf_content)
            board = GoBoard()
            for i in range(len(moves) - 1):
                player, move = moves[i]
                board.apply_move(player, move)
                board_state = board.get_board_layers()
                all_states.append(board_state)
                next_move = moves[i + 1][1]
                all_moves.append(next_move)
    return np.array(all_states), np.array(all_moves)

def build_go_cnn(input_shape=(19, 19, 3)):
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(1024),
        Activation('relu'),
        Dense(361),
        Activation('softmax')
    ])
    return model

def prepare_data(folder_path):
    board_states, next_moves = read_sgf_files(folder_path)
    labels = np.array([ord(move[0]) - ord('a') + 19 * (ord(move[1]) - ord('a')) for move in next_moves])
    labels_one_hot = to_categorical(labels, num_classes=361)
    return train_test_split(board_states, labels_one_hot, test_size=0.2)

# Main execution
folder_path = './Go4Go/'
X_train, X_test, y_train, y_test = prepare_data(folder_path)
model = build_go_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Fit the model with callbacks
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=50, 
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1  # Set verbose to 1 to see the training progress
)

def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()







