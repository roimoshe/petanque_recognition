# petanque_recognition
Final project in Final project of Introduction to Computational and Biological Vision course - petanque recognition
By Roi Moshe and Yuval Rappaport

# Usage
For simple user run, use the following command:
	$ python3 petanque_recognition.py --user
# In order to run more options, use the CLI flags that you can see here:
	$ python3 petanque_recognition.py -h
	usage: petanque_recognition.py [-h] [-s STEP] [-e END_STEP] [-i IMAGE_PATH] [-r RUN_NUM] [-p PLAN_NUM] [-f FRAME] [-q] [-v] [-t] [-c] [-u] [-N] [-F IMAGE_FORMAT]

	Petanque recognition

	optional arguments:
	-h, --help            show this help message and exit
	-s STEP, --step STEP  Input step number to start from
	-e END_STEP, --end_step END_STEP
							Input step number to end in
	-i IMAGE_PATH, --image_path IMAGE_PATH
							Input image number
	-r RUN_NUM, --run_num RUN_NUM
							Input run number
	-p PLAN_NUM, --plan_num PLAN_NUM
							Input plan number
	-f FRAME, --frame FRAME
							Input frame number
	-q, --quick           quick run, without saving any step
	-v, --verbose         verbosity level
	-t, --train           train mode
	-c, --clean           clean build
	-u, --user            user execution
	-N, --no_previous_step
							run from step 'step' with the original photo
	-F IMAGE_FORMAT, --image_format IMAGE_FORMAT
							image format - video/photo/day2