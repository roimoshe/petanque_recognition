# petanque_recognition
Final project in Final project of Introduction to Computational and Biological Vision course - petanque recognition
By Roi Moshe and Yuval Rappaport

# Usage
For simple user run, use the following command:
	$ python3 petanque_recognition.py --user
# In order to run more options, use the CLI flags that you can see here:
	$ python3 petanque_recognition.py -h
	usage: petanque_recognition.py [-h] [-u] [-i IMAGE_PATH] [-v] [-c] [-q] [-s STEP] [-e END_STEP] [-r RUN_NUM] [-p PLAN_NUM] [-f FRAME] [-t] [-N] [-F IMAGE_FORMAT]

	Petanque recognition

	optional arguments:
	  -h, --help            show this help message and exit
	  -u, --user            Simple user execution that dont require any additional flag
	  -i IMAGE_PATH, --image_path IMAGE_PATH
	                        Input image relative path
	  -v, --verbose         verbosity level
	  -c, --clean           clean build
	  -q, --quick           quick run, without saving any step