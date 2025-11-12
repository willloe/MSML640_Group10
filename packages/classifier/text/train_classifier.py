import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class TextTypeClassifierTrainer:
  
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        X_train = np.load(self.data_dir / 'X_train.npy')
        X_test = np.load(self.data_dir / 'X_test.npy')
        y_train = np.load(self.data_dir / 'y_train.npy')
        y_test = np.load(self.data_dir / 'y_test.npy')
        
        with open(self.data_dir / 'label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        with open(self.data_dir / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return X_train, X_test, y_train, y_test, label_mapping, feature_names
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train a classifier model
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'svm'
        """
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"\nTraining {model_type} classifier...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, label_mapping):
        """Evaluate the trained model"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print('='*60)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Reverse label mapping
        id_to_label = {v: k for k, v in label_mapping.items()}
        
        # Get unique labels present in test set
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [id_to_label[i] for i in unique_labels]
        
        print(f"\nClasses present in test set: {len(unique_labels)} out of {len(label_mapping)}")
        print(f"Classes: {', '.join(target_names)}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
        
        # Confusion matrix (only for classes present in test set)
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'unique_labels': unique_labels
        }
    
    def plot_confusion_matrix(self, cm, label_mapping, output_path, unique_labels=None):
        """Plot and save confusion matrix"""
        id_to_label = {v: k for k, v in label_mapping.items()}
        
        # Use only labels present in the data
        if unique_labels is not None:
            labels = [id_to_label[i] for i in unique_labels]
        else:
            labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_names, output_path, top_n=15):
        """Plot feature importance (for tree-based models)"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
        plt.close()
    
    def save_model(self, output_dir):
        """Save the trained model and scaler"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        joblib.dump(self.model, output_dir / 'classifier_model.pkl')
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        
        print(f"\nModel saved to {output_dir}")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'training_data'
    model_dir = script_dir / 'trained_models'
    plots_dir = script_dir / 'plots'
    
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize trainer
    trainer = TextTypeClassifierTrainer(data_dir)
    
    # Load data
    X_train, X_test, y_train, y_test, label_mapping, feature_names = trainer.load_data()
    
    # Train model (try different models)
    model_type = 'random_forest'  # or 'gradient_boosting' or 'svm'
    trainer.train_model(X_train, y_train, model_type=model_type)
    
    # Evaluate
    results = trainer.evaluate_model(X_test, y_test, label_mapping)
    
    # Plot results
    trainer.plot_confusion_matrix(
        results['confusion_matrix'], 
        label_mapping,
        plots_dir / 'confusion_matrix.png',
        unique_labels=results['unique_labels']
    )
    
    trainer.plot_feature_importance(
        feature_names,
        plots_dir / 'feature_importance.png'
    )
    
    # Save model
    trainer.save_model(model_dir)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print('='*60)


if __name__ == '__main__':
    main()