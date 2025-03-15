# main.py
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'notebooks'))

from data_loader import load_data
from train import train_model, SimpleCNN

def main():
    train_dir = "/projects/dsci410_510/Final_proj_ms/train"
    valid_dir = "/projects/dsci410_510/Final_proj_ms/valid"
    test_dir = "/projects/dsci410_510/Final_proj_ms/test"

    num_epochs = 10
    batch_size = 5
    learning_rate = 0.001
    save_path = "trained_model.pth"  # Specify the model save path

    # Load data
    train_loader, valid_loader, test_loader = load_data(train_dir, valid_dir, test_dir, batch_size)

    # Initialize model
    model = SimpleCNN(num_classes=2)

    # Start training
    print("Starting training...")
    train_model(model, train_loader, valid_loader, num_epochs=num_epochs, learning_rate=learning_rate, save_path=save_path)
    print("Training complete.")

if __name__ == "__main__":
    main()

