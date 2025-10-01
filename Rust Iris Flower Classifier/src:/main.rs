//! # Rust Iris Classifier
//!
//! This application provides a full pipeline for a machine learning project in Rust.
//! It includes data fetching, model training, evaluation, and a CLI for predictions.

use std::env;
use std::error::Error;
use ndarray::array;

/// The main entry point of the application.
/// It parses command-line arguments to either train a new model or make a prediction.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            println!("[INFO] Starting model training process...");
            // Run the full training and evaluation pipeline
            let (dataset, labels) = data::get_dataset()?;
            let (model, validation_metrics) = model::train_and_evaluate(&dataset, &labels)?;
            
            println!("[INFO] Training complete.");
            println!("[INFO] Evaluating model...");
            println!("[INFO] Accuracy: {}", validation_metrics.accuracy());
            println!("[INFO] Confusion Matrix:\n{}", validation_metrics.confusion_matrix());

            // Save the trained model
            model::save_model(&model, "iris_model.bin")?;
            println!("[INFO] Model saved to iris_model.bin");
            
            // Generate and save a visualization of the confusion matrix
            visualization::plot_confusion_matrix(validation_metrics.confusion_matrix())?;
            println!("[INFO] Confusion matrix plot saved to confusion_matrix.png");
        }
        Some("predict") => {
            // Check for the correct number of feature arguments
            if args.len() != 6 {
                eprintln!("[ERROR] Usage: cargo run -- predict <sepal_length> <sepal_width> <petal_length> <petal_width>");
                return Ok(());
            }
            
            // Load the pre-trained model
            let model = model::load_model("iris_model.bin")?;
            let (_, labels) = data::get_dataset()?; // We need the original labels for mapping
            let unique_labels = labels.labels();
            
            // Parse the float arguments from the command line
            let features: Result<Vec<f64>, _> = args[2..6].iter().map(|s| s.parse()).collect();
            let features = match features {
                Ok(f) => f,
                Err(_) => {
                    eprintln!("[ERROR] Please provide four valid numbers for the features.");
                    return Ok(());
                }
            };
            
            // Create an ndarray for the prediction
            let new_sample = array(features).into_shape((1, 4))?;
            
            // Make the prediction
            let prediction = model.predict(&new_sample);
            let predicted_label_index = prediction[0];
            let predicted_label = unique_labels[predicted_label_index];
            
            println!("[INFO] Prediction for input {:?}: {}", new_sample.row(0), predicted_label);
        }
        _ => {
            // Print usage instructions if the command is invalid
            eprintln!("[ERROR] Invalid command.");
            eprintln!("Usage:");
            eprintln!("  cargo run -- train          - To train and save a new model.");
            eprintln!("  cargo run -- predict <f1 f2 f3 f4> - To make a prediction with a trained model.");
        }
    }

    Ok(())
}

/// ## `data` Module
/// Handles all data loading and preprocessing tasks.
mod data {
    use polars::prelude::*;
    use std::error::Error;
    use linfa::prelude::*;
    use ndarray::{Array2, Array1};

    /// Downloads and processes the Iris dataset.
    ///
    /// Returns a tuple containing the processed Linfa `Dataset` and the original `LabelSet`.
    pub fn get_dataset() -> Result<(DatasetBase<Array2<f64>, Array1<usize>>, LabelSet), Box<dyn Error>> {
        // Download the dataset from a URL
        let url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv";
        let df = CsvReader::new(reqwest::blocking::get(url)?).finish()?;

        // Define feature column names
        let feature_names = ["sepal.length", "sepal.width", "petal.length", "petal.width"];
        // Extract features into an ndarray
        let features = df.select(feature_names)?.to_ndarray::<Float64Type>()?;

        // Extract and encode the target variable (variety)
        let targets_series = df.column("variety")?.utf8()?;
        // Create a set of unique labels for encoding
        let mut label_set = LabelSet::new();
        let targets: Array1<usize> = targets_series
            .into_iter()
            .filter_map(|opt_label| opt_label.map(|label| label_set.get_or_insert(label)))
            .collect();
            
        // Create a Linfa dataset
        let dataset = Dataset::new(features, targets)
            .with_feature_names(feature_names.to_vec());
            
        Ok((dataset, label_set))
    }
}


/// ## `model` Module
/// Contains logic for model training, evaluation, saving, and loading.
mod model {
    use linfa::prelude::*;
    use linfa_svm::{Svm, Kernel};
    use ndarray::{Array2, Array1};
    use std::fs::File;
    use std::io::{Read, Write};
    use bincode::{serialize, deserialize};
    use std::error::Error;

    /// Trains an SVM model and evaluates its performance on a validation set.
    ///
    /// # Arguments
    /// * `dataset` - The full dataset for training.
    /// * `labels` - The set of unique labels.
    ///
    /// # Returns
    /// A tuple containing the trained model and its validation metrics.
    pub fn train_and_evaluate(
        dataset: &DatasetBase<Array2<f64>, Array1<usize>>,
        _labels: &LabelSet,
    ) -> Result<(Svm<f64, usize>, ConfusionMatrix<usize>), Box<dyn Error>> {
        // Split the dataset into training and validation sets (80/20 split)
        let (train, valid) = dataset.split_with_ratio(0.8);

        // Configure and train the Support Vector Machine (SVM) model
        let model = Svm::params()
            .kernel(Kernel::linear())
            .train(&train)?;

        // Make predictions on the validation set
        let predictions = model.predict(&valid);
        
        // Calculate the confusion matrix to evaluate performance
        let cm = predictions.confusion_matrix(&valid)?;
        
        Ok((model, cm))
    }

    /// Serializes and saves the trained model to a file.
    pub fn save_model(model: &Svm<f64, usize>, path: &str) -> Result<(), Box<dyn Error>> {
        let encoded = serialize(model)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Loads a pre-trained model from a file.
    pub fn load_model(path: &str) -> Result<Svm<f64, usize>, Box<dyn Error>> {
        let mut file = File::open(path)?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)?;
        let model: Svm<f64, usize> = deserialize(&encoded)?;
        Ok(model)
    }
}


/// ## `visualization` Module
/// Responsible for creating plots and other visualizations.
mod visualization {
    use plotters::prelude::*;
    use linfa::prelude::*;
    use std::error::Error;

    /// Creates and saves a heatmap visualization of the confusion matrix.
    pub fn plot_confusion_matrix(cm: &ConfusionMatrix<usize>) -> Result<(), Box<dyn Error>> {
        let labels = cm.labels();
        let n_labels = labels.len();
        let matrix_data = cm.matrix();

        let path = "confusion_matrix.png";
        let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find the max value for color scaling
        let max_val = matrix_data.iter().cloned().fold(0, |acc, v| acc.max(v));

        let mut chart = ChartBuilder::on(&root)
            .caption("Confusion Matrix", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..n_labels as i32, (0..n_labels as i32).rev())?;

        chart.configure_mesh()
            .x_labels(n_labels)
            .y_labels(n_labels)
            .x_label_formatter(&|&x| labels[x as usize].to_string())
            .y_label_formatter(&|&y| labels[y as usize].to_string())
            .draw()?;

        chart.draw_series(
            (0..n_labels).flat_map(|y| (0..n_labels).map(move |x| (x as i32, y as i32, matrix_data[(y, x)])))
                .map(|(x, y, v)| {
                    let alpha = v as f64 / max_val as f64;
                    let color = HSLColor(240.0/360.0, 1.0, 0.5).mix(&WHITE, 1.0 - alpha);
                    let mut rect = Rectangle::new([(x, y), (x + 1, y + 1)], color.filled());
                    rect.set_margin(1, 1, 1, 1);
                    rect
                })
        )?;
        
        // Draw the text values on the heatmap
         chart.draw_series(
            (0..n_labels).flat_map(|y| (0..n_labels).map(move |x| (x as i32, y as i32, matrix_data[(y, x)])))
                .map(|(x, y, v)| {
                    let pos = (x,y);
                     EmptyElement::at(pos)
                        + Text::new(format!("{}", v), (25, 18), ("sans-serif", 20).into_font())
                })
        )?;

        root.present()?;
        Ok(())
    }
}
