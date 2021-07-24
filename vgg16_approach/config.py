import os

base_path = os.path.abspath(os.path.join(os.getcwd(),'../prepare_dataset'))
annots_path = os.path.join(base_path, 'all_informations_normalized.csv')
annots_test_path = os.path.join(base_path, 'test_dataset', 'all_informations_normalized_test.csv')


base_output = 'output'
model_path = os.path.join(base_output, 'detector.h5')
loss_plot_path = os.path.join(base_output, 'losses.png')
accuracy_plot_path = os.path.join(base_output, 'accuracy.png')
