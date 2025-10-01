Rust Iris Flower Classifier
This project is a complete, production-ready example of a machine learning project in Rust. It demonstrates how to build a classifier for the famous Iris flower dataset using the linfa machine learning framework.

🎯 Project Objective
The goal is to build a model that can classify an Iris flower into one of three species (setosa, versicolor, or virginica) based on the length and width of its sepals and petals.

This serves as a template for a typical data science workflow in Rust, including:

Data loading and preprocessing.

Exploratory data analysis (EDA) and visualization.

Model training and validation.

Model serialization (saving/loading).

A simple command-line interface (CLI) for making predictions on new data.

📊 Dataset
The project uses the Iris flower dataset, a classic dataset in machine learning. It contains 150 samples from three species of Iris flowers. Four features were measured for each sample:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

The dataset is downloaded automatically from a remote URL, so no manual download is required.

🛠️ Project Structure
.
├── Cargo.toml      # Project dependencies and metadata
├── README.md       # This file
├── .gitignore      # Standard Rust gitignore
└── src/
    └── main.rs     # Main application logic with modules for data, model, and viz

🚀 Getting Started
Prerequisites

Rust programming language (latest stable version)

A C compiler (like GCC) and pkg-config for some dependencies.

On Debian/Ubuntu: sudo apt-get install build-essential pkg-config

On macOS: xcode-select --install

Installation & Running

Clone the repository:

git clone <your-repo-url>
cd <your-repo-name>

Train the Model:
This command will download the data, train the SVM model, evaluate its performance, save the trained model to iris_model.bin, and generate a confusion matrix plot at confusion_matrix.png.

cargo run --release -- train

Make a Prediction:
Once the model is trained, you can use the predict command to classify a new flower. Provide the sepal length, sepal width, petal length, and petal width as arguments.

# Example prediction
cargo run --release -- predict 5.1 3.5 1.4 0.2

This should predict the species as "Iris-setosa".

📈 Results
The model is evaluated on a held-out test set (20% of the data). The performance metrics are printed to the console during the training run.

Example Output:

[INFO] Training complete.
[INFO] Evaluating model...
[INFO] Accuracy: 1.0
[INFO] Confusion Matrix:
/-------------------\
|  12 |   0 |   0 |
|-------------------|
|   0 |  11 |   0 |
|-------------------|
|   0 |   0 |   7 |
\-------------------/
[INFO] Model saved to iris_model.bin
[INFO] Confusion matrix plot saved to confusion_matrix.png

The high accuracy is typical for this dataset with an SVM model.

(Note: This image will be generated when you run the training command.)

