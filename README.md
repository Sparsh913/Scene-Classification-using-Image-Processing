# Scene-Classification-using-Image-Processing
Basic ML models can classify a scene as a building, a park, a kitchen, and so on. However, in this project, I tried to do it by applying the basic principles of Image Processing.
The overall goal of this exercise is to classify the scenes into the following buckets:
-Aquarium
-Desert
-Highway
-Kitchen
-Laundromat
-Park
-Waterfall
-Windmill

Approach followed: Bag of Visual words
The text classification methods have inspired this approach. Just like we can classify a passage among various topics by examining the number of "related" keywords. But what about an image? What can we examine from an image?
Note that an image is nothing but a matrix of numbers. Various kinds of filters have different types of responses when convolved with an image. Surprisingly, "similar" kind of images responds "similarly" to a filter. Therefore filter banks were used, and responses of every image were stored. Applying K-means clustering to these responses helped visualize the "visual" bag of words.

Code files:
opts: Contains some essential parameters and hyperparameters used throughout the python scripts
util: some helper functions that were often used in the other scripts
visual_words: List of functions aiming to get a dictionary of "visual words"
visual_recog: Using the dictionary made in visual_words, a scene recognition system was built
main: main file calling all the functions for implementation

Read Overview.pdf for in-depth information and literature for this approach.
