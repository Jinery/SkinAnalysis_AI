import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

import utils as ut

def plot_training_history(history, save_path: str = None):

    if save_path is None:
        save_path = os.path.join(ut.get_visualizations_path(), f"training_history.png")

    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history

    epochs = len(history_dict.get('accuracy', history_dict.get('acc', [])))
    epoch_range = range(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    apply_training_history_style(fig, (ax1, ax2))

    ax1.plot(epoch_range, history_dict.get('accuracy', history_dict.get('acc', [])), label='Training Accuracy',
             linewidth=2.5, marker='o', markersize=3)
    ax1.plot(epoch_range, history_dict.get('val_accuracy', history_dict.get('val_acc', [])), label='Validation Accuracy',
             linewidth=2.5, marker='s', markersize=3)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'Epoch (Total: {epochs})', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10, framealpha=0.7)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(1, epochs + 1, max(1, epochs // 10)))

    ax2.plot(epoch_range, history_dict['loss'], label='Training Loss',
             linewidth=2.5, marker='o', markersize=3)
    ax2.plot(epoch_range, history.history['val_loss'], label='Validation Loss',
             linewidth=2.5, marker='s', markersize=3)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'Epoch (Total: {epochs})', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.7)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(range(1, epochs + 1, max(1, epochs // 10)))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Training history saved to {save_path}")
    plt.show()

def apply_training_history_style(fig, axes):
    neon_colors = {
        'training': '#00FF9D',
        'validation': '#FF00FF',
        'bg': '#0A0A14',
        'grid': '#1A1A2E'
    }

    fig.patch.set_facecolor(neon_colors['bg'])
    fig.patch.set_alpha(0.95)

    for ax in axes:
        ax.set_facecolor('#0A0A14')

        lines = ax.get_lines()
        if len(lines) > 0:
            lines[0].set_color(neon_colors['training'])
            lines[0].set_linewidth(2.5)
        if len(lines) > 1:
            lines[1].set_color(neon_colors['validation'])
            lines[1].set_linewidth(2.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444466')
        ax.spines['bottom'].set_color('#444466')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        ax.title.set_color('#FFFFFF')
        ax.xaxis.label.set_color('#CCCCEE')
        ax.yaxis.label.set_color('#CCCCEE')
        ax.tick_params(axis='x', colors='#8888AA')
        ax.tick_params(axis='y', colors='#8888AA')

        ax.grid(True, alpha=0.15, linestyle='--', color='#444466')

        for line in ax.lines:
            line.set_alpha(0.9)


def plot_prediction_matrix(model, val_ds, class_names, history=None):
    y_pred = model.predict(val_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    y_true_classes = np.argmax(y_true, axis=1)

    if history:
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='royalblue')
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='darkorange')
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='royalblue')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='darkorange')
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor('#f8f9fa')

        cm = confusion_matrix(y_true_classes, y_pred_classes)

        ax3 = fig.add_subplot(gs[0, 2])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Normalized Rate'}, ax=ax3)
        ax3.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Predicted Label', fontsize=12)
        ax3.set_ylabel('True Label', fontsize=12)
        ax3.set_facecolor('#f8f9fa')

        ax4 = fig.add_subplot(gs[1, :])

        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, ax=ax4,
                    linewidths=0.5, linecolor='gray')

        ax4.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Predicted Label', fontsize=14, fontweight='semibold')
        ax4.set_ylabel('True Label', fontsize=14, fontweight='semibold')
        ax4.set_facecolor('#f8f9fa')

        plt.suptitle('Model Performance Analysis', fontsize=20, fontweight='bold', y=1.02)

    else:
        fig = plt.figure(figsize=(12, 10))

        cm = confusion_matrix(y_true_classes, y_pred_classes)

        ax = plt.gca()
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    linewidths=0.5, linecolor='gray',
                    annot_kws={"size": 11, "weight": "bold"})

        plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='semibold')
        plt.ylabel('True Label', fontsize=14, fontweight='semibold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    report = classification_report(y_true_classes, y_pred_classes,
                                   target_names=class_names, digits=3)
    print(report)

    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)