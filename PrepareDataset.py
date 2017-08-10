from utils import *
download('http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip', 'data/GTSRB_Final_Training_Images.zip')
#download('http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip', 'data/GTSRB_Final_Test_Images.zip')
# Get the features and labels from the zip files
images, labels = uncompress_features_labels('data/GTSRB_Final_Training_Images.zip',
                                                          'data/','data/GTSRB/Final_Training/Images')

# Wait until you see that all features and labels have been uncompressed.
print('All images and labels uncompressed.')


