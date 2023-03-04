Project Members:
1. Jyothi Vishnu Vardhan Kolla-> Neuid:002752854
2. Vidya Ganesh->NUID:002766414

* Number of time-travel days used -> 3 days.
1.Operating System used: Mac Os
2.Ide Used: CLion


Instructions for Running Executables:

* All the Configurations related to the Project are setup in the File CMakeLists.txt, inorder to execute the code The project
must be opened in the Clion Ide which is already configured with a C++ 17 compiler and add all the .cpp,.h files
into this project and then create and build a new project.

Structure of the Project:
1. imgDisplay.cpp -> File given a image as input performs all the necessary preprocessing operations such as
   thresholding, erosion, dilation, segmentation, least central moments, oriented bounding box and displays
   these effects on images in a window.
2. training.cpp  -> Takes in type of thresholding to perform and then preprocesses all the images in a given
   directory and stores their feature vectors in a csv file.
3. filters.cpp   -> contains functions for all the preprocessing functionalities used in the project like thresholding,
   segmentation, collecting_data etc.
4. classify.cpp  -> this files performs classification given a test directory and a evaluation mode turned on goes through all the
   images in the directory and stores the actual and predicted labels in a csv file.

   * For demonstration we also added the feature where this takes a path to single image and displays the classification result
   in a window.
5. distance_metrics.cpp -> contains the distance measure and functions to perform evaluation for classification tasks.

Extension-1: Adding more Images to the database.

* To test this just run the training.cpp by giving directory of the images.
* To evaluate this run  classify.cpp by setting to evaluation mode(passing yes to 3rd cmd argument).

Extension-2: Enabling recognition of multiple objects using segmentation.

* Run Imgdisplay.cpp by providing the path of the image to test the result will appear in a window.

Extension-3: Experiment with more classifiers/distance metrics.

To test with different distance metrics change the 4th parameter in classify.cpp as following:
1. manhattan_distance: "manhattan_dist"
2. scaled_euclidean  : "scaled_euclidean"
3. chi_square        : "chi-square"

Extension-4: Enable your system to learn unknown objects automatically.
Run the file as shown in the video:
https://drive.google.com/file/d/1Ehl-1PjF0QWdUbG_-f6ovJ4nANlury8A/view?usp=sharing

Extension-5: Explore some object recognition tools in opencv. We explored cascades for face detection
Run the file as shown in the video:
https://drive.google.com/file/d/1nu3h0UZ4ynFR4Pz1tCcqP8cCDSv55Kf5/view?usp=sharing

Domenstration of the system.
https://drive.google.com/file/d/1_jimWT6xyBzeY0CtwC0taBEFOOGhMKcx/view?usp=sharing
https://drive.google.com/file/d/1BG2JacTbi4zyIGuoSN88zbxW7lbnVubP/view?usp=sharing



*** CMAKE configuration for the project.
cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 2.8)
project(main.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main.cpp main.cpp filters.cpp filters.h csv_util.cpp csv_util.h distance_metrics.cpp distance_metrics.h)
add_executable(imgDisplay.cpp imgDisplay.cpp filters.cpp filters.h csv_util.cpp csv_util.h)
add_executable(classify.cpp classify.cpp filters.cpp filters.h csv_util.cpp csv_util.h distance_metrics.cpp distance_metrics.h)
add_executable(training.cpp training.cpp filters.cpp filters.h csv_util.cpp csv_util.h)
add_executable(dnn.cpp dnn.cpp)
target_link_libraries(main.cpp ${OpenCV_LIBS})
target_link_libraries(imgDisplay.cpp ${OpenCV_LIBS})
target_link_libraries(classify.cpp ${OpenCV_LIBS})
target_link_libraries(training.cpp ${OpenCV_LIBS})
target_link_libraries(dnn.cpp ${OpenCV_LIBS})

