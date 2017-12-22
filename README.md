# Emovere-ML

Artificial Intelligence is the science of 21st century. Artificial Intelligence (AI) is defined as the ability for a machine to “think or act humanly or rationally”. Machines are now able to process vast amount of data in real time and respond accordingly. But these machines with high IQ (Intelligence Quotient) were always lacking Emotional Intelligence (EI)/ Emotional Quotient (EQ). As technology progresses and the world becomes more and more virtual, there is a fear that we will lose the human connection and communication; but what if our devices could replace those interactions? The question of the era is whether we can build machines that can recognise human emotions.
Developers and researchers have been advancing artificial intelligence to not only create systems that think and act like humans, but also detect and react to human emotions. Humans show universal consistency in recognizing emotions but also show a great deal of variability between individuals in their abilities. Enabling the devices around us to recognize our emotions can only enhance our interaction with machines, as well as among the family of humanity. The point of this project research is to develop personalized user experiences that can help improve lives. 
With recent advancements in this fields and open source tools such as Tensorflow from Google, creating and training a model is not much difficult. One of the major challenge is to collect the dataset to train the model. Then we came across an Emotion recognition challenge that was hosted by Kaggle “Challenges in Representation Learning: Facial Expression Recognition Challenge”. 56 Teams participated in this challenge and different approaches were used including Haar, Hog, SIFT, neural networks etc. The dataset given for this challenge is publicly available to download known as FER2013. 
The dataset was created using the Google image search API to search for images of faces that match a set of 184 emotion-related keywords like “blissful”, “enraged,” etc. These keywords were combined with words related to gender, age or ethnicity, to obtain nearly 600 strings, which were used as facial image search queries. The first 1000 images returned for each query were kept for the next stage of processing. OpenCV face recognition was used to obtain bounding boxes around each face in the collected images. Human labellers than rejected incorrectly labelled images, corrected the cropping if necessary, and filtered out some duplicate images. Approved, cropped images were then resized to 48x48 pixels and converted to grayscale. The resulting dataset contains 35887 images, with 4953 “Anger” images, 547 “Disgust” images, 5121 “Fear” images, 8989 “Happiness” images, 6077 “Sadness” images, 4002 “Surprise” images, and 6198 “Neutral” images. 
Incorporating recent advancements in the neural networks, a model was implemented using Tensorflow and Keras. It is a light weight model that works in real time, so that it can be used even on hardware constrained systems.

The model achieved 66% accuracy on 94th epoch with batch size 32.
--Will Be Updated Soon--

Special Thanks & Credits to:
Jostine Ho(https://github.com/JostineHo)
Octavio Arriaga(https://github.com/oarriaga)