#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.81.03), Wed 04 Feb 2015 11:22:15 AM EST
If you publish work using this script please cite the relevant PsychoPy publications
  Peirce, JW (2007) PsychoPy - Psychophysics software in Python. Journal of Neuroscience Methods, 162(1-2), 8-13.
  Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy. Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, core, data, event, logging, sound, gui
from psychopy.constants import *  # things like STARTED, FINISHED
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import csv
import math
import pandas as pd
from psychopy.hardware import keyboard

from murfi_activation_communicator import MurfiActivationCommunicator

# button box
left_button='1'
right_button='2'
enter_button='3'

BaselineFrames=25
exp_tr=1.2
#std_factor=1.6

def get_sma(prices, rate):
    return prices.rolling(rate).mean()
    

def get_bollinger_bands(prices, rate):
    sma = get_sma(prices, rate) # <-- Get SMA for 20 days
    std = prices.rolling(rate).std() # <-- Get rolling standard deviation for 20 days
    CEN_trigger = sma + std * std_factor # Calculate top band
    DMN_trigger = sma - std * std_factor # Calculate bottom band
    return CEN_trigger, DMN_trigger



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

##################################################################################
## PARSE COMMAND LINE ARGUMENTS TO AUTOFILL DIALOGUE BOX AT STARTUP (for runs 2+) ##
num_cmd_line_arguments = len(sys.argv)
if num_cmd_line_arguments >= 2:
    input_participant = sys.argv[1]
else:
    input_participant = ''

# cmd line arg 3 will be ses number
if num_cmd_line_arguments >= 3:
    input_session = sys.argv[2]
else:
    input_session = ''

# cmd line arg 4 will be run number
if num_cmd_line_arguments >= 4:
    input_run = sys.argv[3]
else:
    input_run = ''

# These inputs should be pre-determined, not by user each time
input_noROIs = '2';
input_level = '1';
input_noRepetitions = '1';
input_runTime = '250';
input_scaleFactor = '10';
input_stdFactor = '2.0';
input_session = 'feedback';
# Store info about the experiment 
expName = 'dmnelf'  # from the Builder filename that created this script
expInfo = {'participant':input_participant,'session':input_session,'run':input_run,'No_of_ROIs':input_noROIs,\
'Level_1_2_3':input_level,'No_repetitions':input_noRepetitions,'Run_Time':input_runTime,'Scale_Factor':input_scaleFactor,\
'std_factor':input_stdFactor}#Run_Time in seconds and direction  

murfi_FAKE=False

expInfo['No_of_ROIs'] = input_noROIs
expInfo['Level_1_2_3'] = input_level
expInfo['No_repetitions'] = input_noRepetitions
expInfo['Run_Time'] = input_runTime
expInfo['Scale_Factor'] = input_scaleFactor
expInfo['std_factor'] = input_stdFactor

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName, 
        labels = {'participant': 'Participant ID (###)', 
                  'session': 'Sesssion (feedback)',
                  'run': 'Run (1/2)', 
                  'No_of_ROIs': '# of ROIs',
                  'Level_1_2_3': 'Level (1/2/3)',
                  'No_repetitions': '# of repetitions',
                  'Run_Time': 'Run time (# of TRs)',
                  'Scale_Factor': 'Scale factor',
                  'std_factor': 'std factor'},
        order = ['participant', 'session','run','No_of_ROIs','Level_1_2_3','No_repetitions','Run_Time','Scale_Factor','std_factor'])
if dlg.OK == False: core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
roi_number= str('%s') %(expInfo['No_of_ROIs'])
roi_number=int(roi_number)

RUN_TIME= str('%s') %(expInfo['Run_Time'])
#add 5 sec of buffer to finish the readout for writing to scripts
RUN_TIME=int(RUN_TIME)+5
RUN_TIME=RUN_TIME

nReps=str('%s') %(expInfo['No_repetitions'])
nReps=int(nReps)

position_distance=expInfo['Level_1_2_3']
position_distance=int(position_distance)

scale_factor_z2pixels=expInfo['Scale_Factor']
scale_factor_z2pixels=int(scale_factor_z2pixels)

# Setup files for saving
if not os.path.isdir('data'):
    os.makedirs('data')  # if this fails (e.g. permissions) we will get error

if not os.path.exists(f"data/sub-dmnelf{expInfo['participant']}"):
    os.mkdir(f"data/sub-dmnelf{expInfo['participant']}")    
    
filename = 'data' + os.path.sep + 'sub-dmnelf%s' %(expInfo['participant']) + os.path.sep + 'sub-dmnelf%s_ses-%s_task-experiencesampling_run-0%s' %(expInfo['participant'],expInfo['session'],expInfo['run'])
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

with open(filename+'_events.csv', 'a', newline="") as csvfile:
            stim_writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            stim_writer.writerow(['onset','duration','trial_type','frame','cen_activation','dmn_activation','pda_timeseries','pda_moving_average','CEN_trigger_val','DMN_trigger_val','cen_trigger_counter','dmn_trigger_counter','CEN_response','CEN_response_rt','CEN_mood_response','CEN_mood_response_rt','CEN_duration_response','CEN_duration_response_rt','DMN_response','DMN_response_rt','DMN_mood_response','DMN_mood_response_rt','DMN_duration_response','DMN_duration_response_rt'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
#save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(size=(1080,800), fullscr=True, screen=1, allowGUI=False, allowStencil=False,#1080, 1080
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    )
win.mouseVisible = False
# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win.getActualFrameRate()
if expInfo['frameRate']!=None:
    frameDur = 1.0/round(expInfo['frameRate'])
else:
    frameDur = 1.0/60.0 # couldn't get a reliable measure so guess


# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
instructions_text = visual.TextStim(win=win, ori=0, name='instructions_text',
        text=u'In the next 6 minutes, please gaze softly at the cross on the \nscreen while letting your thoughts flow freely. \nBlink naturaly but please do not close your eyes for a longer period. \n\nFrom time to time, you may see an exclamation mark (!) appear on \nthe screen for a second. At that moment, please reflect on what you were \njust now thinking. \n\nNext, use the slider to answer two questions about your thoughts. \nPlease use your index finger to move it to the left \nand the middle finger to move to the right.\n\nThere are no right or wrong answers to this questions,\njust be as honest and accurate as possible \nwhile reflecting on your thoughts and feelings.\n\nPlease press any key to continue.',
        font=u'Arial',
    pos=[0, 0], height=0.08, wrapWidth=2,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "trigger"
triggerClock = core.Clock()
trigger_text = visual.TextStim(win=win, ori=0, name='trigger_text',
    text=u'waiting for scanner to start...',    font=u'Arial',
    pos=[0, 0], height=0.1, wrapWidth=2,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "baseline"
baselineClock = core.Clock()
mainClock = core.Clock()
baseline_text = visual.TextStim(win=win, ori=0, name='baseline_text',
    text=u'+',    font=u'Arial',
    pos=[0, 0], height=0.3, wrapWidth=None,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0)
    
# Initialize components for Routine "experience_sampling_exclamation"
experience_sampling_exclamationClock = core.Clock()
text_experience_sampling_exclamation = visual.TextStim(win=win, ori=0, name='text_experience_sampling_exclamation',
    text=u'!',    font=u'Arial',
    pos=[0, 0], height=0.8, wrapWidth=2,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0)

def run_slider(question_text='Default Text', left_label='left', right_label='right'):
    slider_question = visual.TextStim(win=win, ori=0, name='text',
        text=question_text, font=u'Arial',
        pos=[0, 0.2], height=0.1, wrapWidth=1.2,
        color=u'white', colorSpace='rgb', opacity=1,
        depth=0.0)

    vas = visual.Slider(win,
                size=(0.85, 0.1),
                ticks=(1, 9),
                labels=(left_label, right_label),
                granularity=1,
                color='white',
                fillColor='white',
                font=u'Arial')

    event.clearEvents('keyboard')
    vas.markerPos = 5
    vas.draw()
    slider_question.draw()
    win.flip()
    continueRoutine = True
    while continueRoutine:
        keys = event.getKeys(keyList=[left_button, right_button, enter_button])
        if len(keys):
            if left_button in keys:
                vas.markerPos = vas.markerPos - 1
            elif right_button in keys:
                vas.markerPos = vas.markerPos  + 1 
            elif enter_button in keys:
                vas.rating=vas.markerPos
                continueRoutine=False
            vas.draw()
            slider_question.draw()
            win.flip()
            print(keys)

    print(f'Rating: {vas.rating}, RT: {vas.rt}')
    with open(run_questions_file, 'a') as csvfile:
            stim_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            stim_writer.writerow([expInfo['participant'], expInfo['run'], expInfo['feedback_on'],
                                  question_text, vas.rating, vas.rt])   

    
    return(vas.rating)

# Initialize components for Routine "experience_sampling_mood"
experience_sampling_moodClock = core.Clock()
text_experience_sampling_mood = visual.TextStim(win=win, ori=0, name='text_experience_sampling_mood',
    text=u'What was the mood of your last thought？',    font='Arial',
    pos=[0, 0.005], height=0.08, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)
rating_experience_sampling_mood = visual.RatingScale(win=win, name='rating_experience_sampling_mood', marker=u'triangle', size=1.0, pos=[0.0, -0.4], low=1, high=3, labels=[u'pleasant', u'neutral',u'unpleasant'], scale=u'', leftKeys='1', rightKeys = '2',markerStart=u'2', showAccept=True)

# Initialize components for Routine "experience_sampling_duration"
experience_sampling_durationClock = core.Clock()
text_experience_sampling_duration = visual.TextStim(win=win, ori=0, name='text_experience_sampling_duration',
    text=u'How long was your last train of thought?',    font=u'Arial',
    pos=[0, 0.005], height=0.08, wrapWidth=None,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0)
rating_experience_sampling_duration = visual.RatingScale(win=win, name='rating_experience_sampling_duration', marker=u'triangle', size=1.0, pos=[0.0, -0.4], low=1, high=3, labels=[u'<15 seconds', u'15-30 seconds',u'>30 seconds'], scale=u'', leftKeys='1', rightKeys = '2',markerStart=u'2', showAccept=True)


# Ask slider questions
#run_slider(question_text='你的情绪愉悦吗？',
#                left_label='一点都不愉悦', right_label='非常愉悦')
#run_slider(question_text='你最近的一个思绪持续了多久？',
#                left_label='５秒以内', right_label='１分钟以上')

cooldown_counter=0
cen_trigger_counter=0
dmn_trigger_counter=0
frame_post_trigger =0
CEN_response=0
CEN_response_rt=0
CEN_mood_response=0
CEN_mood_response_rt=0
CEN_duration_response=0
CEN_duration_response_rt=0
DMN_response=0
DMN_response_rt=0
DMN_mood_response=0
DMN_mood_response_rt=0
DMN_duration_response=0
DMN_duration_response_rt=0


# Initialize components for Routine "feedback"
feedbackClock = core.Clock()


 #murfi communicator
BaseLineTime=BaselineFrames*exp_tr #30 
onset=BaseLineTime
std_factor=float(expInfo['std_factor'])
print("std_factor:",std_factor)
roi_names = ['cen', 'dmn']#, 'mpfc','wm']
# REPLACE THIS IP WITH THE MURFI COMPUTER'S IP 192.168.2.5
#communicator = MurfiActivationCommunicator('18.111.80.133',
#communicator = MurfiActivationCommunicator('18.189.76.118',
communicator = MurfiActivationCommunicator('192.168.2.5',
                                           15001, RUN_TIME,
                                           roi_names,exp_tr,murfi_FAKE)
print ("murfi communicator ok")


# update component parameters for each repeat
key_resp_feedback = event.BuilderKeyResponse()  # create an object of type KeyResponse
key_resp_feedback.status = NOT_STARTED

roi_names_list=['cen','dmn']
print (roi_names_list)
n_roi = roi_number


# Initialize components for Routine "finish"
finishClock = core.Clock()
thank_you_end_run_text = visual.TextStim(win=win, ori=0, name='thank_you_end_run_text',
    text=u'Thank you！',    font=u'Arial',
    pos=[0, 0], height=0.1, wrapWidth=None,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0)


# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 



#------Prepare to start Routine "instructions"-------
t = 0
instructionsClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
key_resp_2 = event.BuilderKeyResponse()  # create an object of type KeyResponse
key_resp_2.status = NOT_STARTED
# keep track of which components have finished
instructionsComponents = []
instructionsComponents.append(instructions_text)
instructionsComponents.append(key_resp_2)
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instructions"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instructionsClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    if t >= 0.0 and instructions_text.status == NOT_STARTED:
        # keep track of start time/frame for later
        instructions_text.tStart = t  # underestimates by a little under one frame
        instructions_text.frameNStart = frameN  # exact frame index
        instructions_text.setAutoDraw(True)
    
    # *key_resp_2* updates
    if t >= 0.0 and key_resp_2.status == NOT_STARTED:
        # keep track of start time/frame for later
        key_resp_2.tStart = t  # underestimates by a little under one frame
        key_resp_2.frameNStart = frameN  # exact frame index
        key_resp_2.status = STARTED
        # keyboard checking is just starting
        key_resp_2.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if key_resp_2.status == STARTED:
        theseKeys = event.getKeys(keyList=['space','1','2'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            key_resp_2.keys = theseKeys[-1]  # just the last key pressed
            key_resp_2.rt = key_resp_2.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instructions"-------
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_2.keys in ['', [], None]:  # No response was made
   key_resp_2.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('key_resp_2.keys',key_resp_2.keys)
if key_resp_2.keys != None:  # we had a response
    thisExp.addData('key_resp_2.rt', key_resp_2.rt)
thisExp.nextEntry()


#------Prepare to start Routine "trigger"-------
t = 0
triggerClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
key_resp_3 = event.BuilderKeyResponse()  # create an object of type KeyResponse
key_resp_3.status = NOT_STARTED
# keep track of which components have finished
triggerComponents = []
triggerComponents.append(trigger_text)
triggerComponents.append(key_resp_3)
for thisComponent in triggerComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "trigger"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = triggerClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *trigger_text* updates
    if t >= 0.0 and trigger_text.status == NOT_STARTED:
        # keep track of start time/frame for later
        trigger_text.tStart = t  # underestimates by a little under one frame
        trigger_text.frameNStart = frameN  # exact frame index
        trigger_text.setAutoDraw(True)
    
    # *key_resp_3* updates
    if t >= 0.0 and key_resp_3.status == NOT_STARTED:
        # keep track of start time/frame for later
        key_resp_3.tStart = t  # underestimates by a little under one frame
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.status = STARTED
        # keyboard checking is just starting
        key_resp_3.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if key_resp_3.status == STARTED:
        theseKeys = event.getKeys(keyList=['num_add', 't','+','5','s','S'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            key_resp_3.keys = theseKeys[-1]  # just the last key pressed
            key_resp_3.rt = key_resp_3.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in triggerComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "trigger"-------
for thisComponent in triggerComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_3.keys in ['', [], None]:  # No response was made
   key_resp_3.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('key_resp_3.keys',key_resp_3.keys)
if key_resp_3.keys != None:  # we had a response
    thisExp.addData('key_resp_3.rt', key_resp_3.rt)
thisExp.nextEntry()

#------Prepare to start Routine "baseline"-------
t = 0
baselineClock.reset()  # clock 
frameN = -1
routineTimer.add(BaseLineTime)
# update component parameters for each repeat
# keep track of which components have finished
baselineComponents = []
baselineComponents.append(baseline_text)
for thisComponent in baselineComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "baseline"-------
continueRoutine = True
print("starting baseline")
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    communicator.update()
    t = baselineClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *baseline_text* updates
    if t >= 0.0 and baseline_text.status == NOT_STARTED:
        # keep track of start time/frame for later
        baseline_text.tStart = t  # underestimates by a little under one frame
        baseline_text.frameNStart = frameN  # exact frame index
        baseline_text.setAutoDraw(True)
    if baseline_text.status == STARTED and t >= (0.0 + (BaseLineTime-win.monitorFramePeriod*0.75)): #most of one frame period left
        baseline_text.setAutoDraw(False)
        with open(filename+'_events.csv', 'a', newline="") as csvfile:
            stim_writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            stim_writer.writerow(['0',BaseLineTime,'baseline'])#,frame,roi_raw_activations[0],roi_raw_activations[1],pda_timeseries[-1],PDA_moving_average['pda'].iloc[-1],CEN_trigger['pda'].iloc[-1],DMN_trigger['pda'].iloc[-1],cen_trigger_counter,dmn_trigger_counter,CEN_response,CEN_response_rt_rt,CEN_mood_response,CEN_mood_response_rt,CEN_duration_response,CEN_duration_response_rt,0,0,0,0,0,0,0,0])

    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in baselineComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
            
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#-------Ending Routine "baseline"-------
for thisComponent in baselineComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=nReps, method='random', 
    extraInfo=expInfo, originPath=None,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial.keys():
        exec('{} = thisTrial[paramName]'.format(paramName))



#prepare to start routine feedback
#create file to save DMN and CEN activity per TR 
for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial.keys():
            exec('{} = thisTrial[paramName]'.format(paramName))

#------Prepare to start Routine "feedback"-------
    t = 0
    feedbackClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    subject_key_target = event.BuilderKeyResponse()  # create an object of type KeyResponse
    subject_key_target.status = NOT_STARTED
    subject_key_reset = event.BuilderKeyResponse()  # create an object of type KeyResponse
    subject_key_reset.status = NOT_STARTED
    routineTimer.add(RUN_TIME)
    
    frame =  BaselineFrames
    dmn_feedback = []
    #mpfc_feedback = []
    cen_feedback = []
    dmn_mpfc_feedback=[]
    mpfc_cen_feedback=[]
    wm_feedback = []
    times = []
    pda_timeseries =[]
    cen_trigger_counter=0
    dmn_trigger_counter=0
    #-------Start Routine "feedback"-------
    activity=0
    out_of_bounds=position_distance*0.4
    for i in range(n_roi):
        activity_i=0
    continueRoutine = True
    
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = feedbackClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        communicator.update()
        
        roi_raw_activations=[]
        for i in range(n_roi):
            roi_raw_i=communicator.get_roi_activation(roi_names_list[i], frame)
            roi_raw_activations.append(roi_raw_i)

        if roi_raw_activations[0] ==0: #and roi_raw_activations[0]==0:
            #win.close()
            print ("let's begin feedback")
           

        elif roi_raw_activations[0] != roi_raw_activations[0] or roi_raw_activations[0] != roi_raw_activations[0]:
            #print ("began baseline")
            continue
        
        roi_activities=[]
        
        for i in range(n_roi):
            target_roi_i=(roi_raw_activations[i])#-drift_roi_raw) include this if wm mask is used to substract activity
            roi_activities.append(target_roi_i)
        
        #Get Positive Diametric Activity between CEN and DMN rois for trigger calculation
        pda_timeseries.append(roi_raw_activations[0]-roi_raw_activations[1])
        pda_timeseries_df = pd.DataFrame(pda_timeseries, columns=['pda'])
        """target_pcc=(pcc-wm)
        target_mpfc=(mpfc-wm)
        target_dlpfc=(dlpfc-wm)"""
        #print "roi actitivities",roi_activities
        #print ("got feedback at time : ", frame, roi_raw_activations, roi_names_list)
     
        
  
        
        #print frame, "PCC= ",roi_activities[0], "MPFC= ",roi_activities[1], "DLPFC= ", roi_activities[2]
        
        #test for one direction unmark this
        #roi_activities[1]=roi_activities[1]+1
        
        """print "di at time %d: %f, %f, %f, %f" % (frame, pcc, mpfc,dlpfc,wm)
        print frame, "PCC= ",target_pcc, "MPFC= ",target_mpfc, "DLPFC= ", target_dlpfc
        roi_activities=(target_pcc,target_mpfc,target_dlpfc)"""
        
        #print "roi activities", roi_activities
        #print frame, cursor_position
        #print 'max_roi: ',max(roi_activities),'index:',roi_activities.index(max(roi_activities))

        
        #Check for PDA threshold and draw the experience sample question question
        
        times.append(frame)
        frame += 1
        #print(frame,pda_timeseries)
 
        #print(frame,PDA_mean,CEN_trigger,DMN_trigger)
        if frame > BaselineFrames:
            onset+=exp_tr
            PDA_moving_average = get_sma(pda_timeseries_df, BaselineFrames) # Get SMA
            #=moving_average(pda_timeseries,BaselineFrames)
            PDA_moving_stdev=get_bollinger_bands(pda_timeseries_df,BaselineFrames)
            #PDA_moving_stdev=moving_stdev(pda_timeseries, BaselineFrames)
            #PDA_mean = np.mean(pda_timeseries)
            CEN_trigger, DMN_trigger = get_bollinger_bands(pda_timeseries_df,BaselineFrames)

            #CEN_trigger = PDA_moving_average+PDA_moving_stdev
            #DMN_trigger = PDA_moving_average-PDA_moving_stdev
            print(frame)
            #print(type(PDA_moving_average),PDA_moving_average['pda'].iloc[-1]) #dataframe
            #print(type(PDA_moving_stdev),PDA_moving_stdev[-1]) #tuple
            #print('cen',type(CEN_trigger),CEN_trigger['pda'].iloc[-1]) #dataframe
            #print('dmn',type(DMN_trigger),DMN_trigger['pda'].iloc[-1])#dataframe
            if cooldown_counter==0:
                if pda_timeseries[-1] >= CEN_trigger['pda'].iloc[-1]:
                    cooldown_counter=15
                    cen_trigger_counter+=1
                    
                    #------Prepare to start Routine "experience_sampling_exclamation"-------
                    t = 0
                    experience_sampling_exclamationClock.reset()  # clock 
                    frameN = -1
                    routineTimer.add(exp_tr)
                    # update component parameters for each repeat
                    
                    # keep track of which components have finished
                    experience_sampling_exclamationComponents = []
                    #change color of exclamation mark
                    text_experience_sampling_exclamation = visual.TextStim(win=win, text='!',color='white', height=0.3)
                    experience_sampling_exclamationComponents.append(text_experience_sampling_exclamation)
                    for thisComponent in experience_sampling_exclamationComponents:
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED

                    #-------Start Routine "experience_sampling_exclamation"-------
                    continueRoutine = True
                    while continueRoutine and routineTimer.getTime() > 0:
                        # get current time
                        t = experience_sampling_exclamationClock.getTime()
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_experience_sampling_exclamation* updates
                        if t >= 0.0 and text_experience_sampling_exclamation.status == NOT_STARTED:
                            # keep track of start time/frame for later
                            text_experience_sampling_exclamation.tStart = t  # underestimates by a little under one frame
                            text_experience_sampling_exclamation.frameNStart = frameN  # exact frame index
                            text_experience_sampling_exclamation.setAutoDraw(True)
                            
                        elif text_experience_sampling_exclamation.status == STARTED and t >= (0.0 + (exp_tr-win.monitorFramePeriod*0.75)): #most of one frame period left
                            text_experience_sampling_exclamation.setAutoDraw(False)
                            continueRoutine = False

                        # check for quit (the Esc key)
                        if endExpNow or event.getKeys(keyList=["escape"]):
                            core.quit()
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                            
                            

                            
                    #------Prepare to start Routine "experience_sampling_mood"-------
                    t = 0
                    experience_sampling_moodClock.reset()  # clock 
                    frameN = -1
                    routineTimer.add(exp_tr*7)
                    # update component parameters for each repeat
                    rating_experience_sampling_mood.reset()
                    # keep track of which components have finished
                    experience_sampling_moodComponents = []
                    experience_sampling_moodComponents.append(text_experience_sampling_mood)
                    experience_sampling_moodComponents.append(rating_experience_sampling_mood)
                    for thisComponent in experience_sampling_moodComponents:
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED

                    #-------Start Routine "experience_sampling_mood"-------
                    continueRoutine = True
                    while continueRoutine and routineTimer.getTime() > 0:
                        # get current time
                        t = experience_sampling_moodClock.getTime()
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_experience_sampling_mood* updates
                        if t >= 0.0 and text_experience_sampling_mood.status == NOT_STARTED:
                            # keep track of start time/frame for later
                            text_experience_sampling_mood.tStart = t  # underestimates by a little under one frame
                            text_experience_sampling_mood.frameNStart = frameN  # exact frame index
                            text_experience_sampling_mood.setAutoDraw(True)
                            rating_experience_sampling_mood.setAutoDraw(True)
                        elif text_experience_sampling_mood.status == STARTED and t >= (0.0 + (exp_tr*3-win.monitorFramePeriod*0.75)): #most of one frame period left
                            rating_experience_sampling_mood.response = rating_experience_sampling_mood.getRating()
                            rating_experience_sampling_mood.rt = rating_experience_sampling_mood.getRT()
                            print("mood response was:",rating_experience_sampling_mood.getRating())
                            CEN_mood_response = rating_experience_sampling_mood.response
                            CEN_mood_response_rt = rating_experience_sampling_mood.rt
                            text_experience_sampling_mood.setAutoDraw(False)
                            rating_experience_sampling_mood.setAutoDraw(False)
                            continueRoutine = False

                        # check for quit (the Esc key)
                        if endExpNow or event.getKeys(keyList=["escape"]):
                            core.quit()
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()


                    #------Prepare to start Routine "experience_sampling_duration"-------
                    t = 0
                    experience_sampling_durationClock.reset()  # clock 
                    frameN = -1
                    routineTimer.add(exp_tr*7)
                    # update component parameters for each repeat
                    rating_experience_sampling_duration.reset()
                    # keep track of which components have finished
                    experience_sampling_durationComponents = []
                    experience_sampling_durationComponents.append(text_experience_sampling_duration)
                    experience_sampling_durationComponents.append(rating_experience_sampling_duration)
                    for thisComponent in experience_sampling_durationComponents:
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED

                    #-------Start Routine "experience_sampling_duration"-------
                    continueRoutine = True
                    while continueRoutine and routineTimer.getTime() > 0:
                        # get current time
                        t = experience_sampling_durationClock.getTime()
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_experience_sampling_duration* updates
                        if t >= 0.0 and text_experience_sampling_duration.status == NOT_STARTED:
                            # keep track of start time/frame for later
                            text_experience_sampling_duration.tStart = t  # underestimates by a little under one frame
                            text_experience_sampling_duration.frameNStart = frameN  # exact frame index
                            text_experience_sampling_duration.setAutoDraw(True)
                            rating_experience_sampling_duration.setAutoDraw(True)
                        elif text_experience_sampling_duration.status == STARTED and t >= (0.0 + (exp_tr*3-win.monitorFramePeriod*0.75)): #most of one frame period left
                            rating_experience_sampling_duration.response = rating_experience_sampling_duration.getRating()
                            rating_experience_sampling_duration.rt = rating_experience_sampling_duration.getRT()
                            print("duration response was:",rating_experience_sampling_duration.getRating())
                            CEN_duration_response = rating_experience_sampling_duration.response
                            CEN_duration_response_rt = rating_experience_sampling_duration.rt
                            text_experience_sampling_duration.setAutoDraw(False)
                            rating_experience_sampling_duration.setAutoDraw(False)
                            #text_4 = visual.TextStim(win=win, text='+',color='red', height=0.3)
                            continueRoutine = False

                        # check for quit (the Esc key)
                        if endExpNow or event.getKeys(keyList=["escape"]):
                           core.quit()
                            
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                           win.flip()


                    else:
                        text_4 = visual.TextStim(win=win, text='+',color='white', height=0.3)

                elif pda_timeseries[-1] <= DMN_trigger['pda'].iloc[-1]:
                    cooldown_counter=15
                    dmn_trigger_counter+=1
                    #text_4 = visual.TextStim(win=win, text='+',color='blue', height=0.3)
                    #------Prepare to start Routine "experience_sampling_exclamation"-------
                    t = 0
                    experience_sampling_exclamationClock.reset()  # clock 
                    frameN = -1
                    routineTimer.add(exp_tr)
                    # update component parameters for each repeat
                    # keep track of which components have finished
                    experience_sampling_exclamationComponents = []
                    #change color of exclamation mark
                    text_experience_sampling_exclamation = visual.TextStim(win=win, text='!',color='white', height=0.3)
                    experience_sampling_exclamationComponents.append(text_experience_sampling_exclamation)
                    for thisComponent in experience_sampling_exclamationComponents:
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED

                    #-------Start Routine "experience_sampling_exclamation"-------
                    continueRoutine = True
                    while continueRoutine and routineTimer.getTime() > 0:
                        # get current time
                        t = experience_sampling_exclamationClock.getTime()
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_experience_sampling_exclamation* updates
                        if t >= 0.0 and text_experience_sampling_exclamation.status == NOT_STARTED:
                            # keep track of start time/frame for later
                            text_experience_sampling_exclamation.tStart = t  # underestimates by a little under one frame
                            text_experience_sampling_exclamation.frameNStart = frameN  # exact frame index
                            text_experience_sampling_exclamation.setAutoDraw(True)
                            
                        elif text_experience_sampling_exclamation.status == STARTED and t >= (0.0 + (exp_tr-win.monitorFramePeriod*0.75)): #most of one frame period left
                            #print(type(CEN_response),CEN_response)
                            text_experience_sampling_exclamation.setAutoDraw(False)
                            continueRoutine = False

                        # check for quit (the Esc key)
                        if endExpNow or event.getKeys(keyList=["escape"]):
                            core.quit()
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                            
                            

                            
                    #------Prepare to start Routine "experience_sampling_mood"-------
                    t = 0
                    experience_sampling_moodClock.reset()  # clock 
                    frameN = -1
                    routineTimer.add(exp_tr*7)
                    # update component parameters for each repeat
                    rating_experience_sampling_mood.reset()
                    # keep track of which components have finished
                    experience_sampling_moodComponents = []
                    experience_sampling_moodComponents.append(text_experience_sampling_mood)
                    experience_sampling_moodComponents.append(rating_experience_sampling_mood)
                    for thisComponent in experience_sampling_moodComponents:
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED

                    #-------Start Routine "experience_sampling_mood"-------
                    continueRoutine = True
                    while continueRoutine and routineTimer.getTime() > 0:
                        # get current time
                        t = experience_sampling_moodClock.getTime()
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_experience_sampling_mood* updates
                        if t >= 0.0 and text_experience_sampling_mood.status == NOT_STARTED:
                            # keep track of start time/frame for later
                            text_experience_sampling_mood.tStart = t  # underestimates by a little under one frame
                            text_experience_sampling_mood.frameNStart = frameN  # exact frame index
                            text_experience_sampling_mood.setAutoDraw(True)
                            rating_experience_sampling_mood.setAutoDraw(True)
                        elif text_experience_sampling_mood.status == STARTED and t >= (0.0 + (exp_tr*3-win.monitorFramePeriod*0.75)): #most of one frame period left
                            rating_experience_sampling_mood.response = rating_experience_sampling_mood.getRating()
                            rating_experience_sampling_mood.rt = rating_experience_sampling_mood.getRT()
                            print("mood was:",rating_experience_sampling_mood.getRating())
                            DMN_mood_response = rating_experience_sampling_mood.response
                            DMN_mood_response_rt = rating_experience_sampling_mood.rt
                            text_experience_sampling_mood.setAutoDraw(False)
                            rating_experience_sampling_mood.setAutoDraw(False)
                            continueRoutine = False

                        # check for quit (the Esc key)
                        if endExpNow or event.getKeys(keyList=["escape"]):
                            core.quit()
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()


                    #------Prepare to start Routine "experience_sampling_duration"-------
                    t = 0
                    experience_sampling_durationClock.reset()  # clock 
                    frameN = -1
                    routineTimer.add(exp_tr*7)
                    # update component parameters for each repeat
                    rating_experience_sampling_duration.reset()
                    # keep track of which components have finished
                    experience_sampling_durationComponents = []
                    experience_sampling_durationComponents.append(text_experience_sampling_duration)
                    experience_sampling_durationComponents.append(rating_experience_sampling_duration)
                    for thisComponent in experience_sampling_durationComponents:
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED

                    #-------Start Routine "experience_sampling_duration"-------
                    continueRoutine = True
                    while continueRoutine and routineTimer.getTime() > 0:
                        # get current time
                        t = experience_sampling_durationClock.getTime()
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_experience_sampling_duration* updates
                        if t >= 0.0 and text_experience_sampling_duration.status == NOT_STARTED:
                            # keep track of start time/frame for later
                            text_experience_sampling_duration.tStart = t  # underestimates by a little under one frame
                            text_experience_sampling_duration.frameNStart = frameN  # exact frame index
                            text_experience_sampling_duration.setAutoDraw(True)
                            rating_experience_sampling_duration.setAutoDraw(True)
                        elif text_experience_sampling_duration.status == STARTED and t >= (0.0 + (exp_tr*3-win.monitorFramePeriod*0.75)): #most of one frame period left
                            rating_experience_sampling_duration.response = rating_experience_sampling_duration.getRating()
                            rating_experience_sampling_duration.rt = rating_experience_sampling_duration.getRT()
                            print("duration was:",rating_experience_sampling_duration.getRating())
                            DMN_duration_response = rating_experience_sampling_duration.response
                            DMN_duration_response_rt = rating_experience_sampling_duration.rt
                            text_experience_sampling_duration.setAutoDraw(False)
                            rating_experience_sampling_duration.setAutoDraw(False)
                            #text_4 = visual.TextStim(win=win, text='+',color='red', height=0.3)
                            continueRoutine = False

                        # check for quit (the Esc key)
                        if endExpNow or event.getKeys(keyList=["escape"]):
                           core.quit()
                            
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                           win.flip()


                    else:
                        text_4 = visual.TextStim(win=win, text='+',color='white', height=0.3)
                    
                    
                    
                else:
                    cooldown_counter=cooldown_counter
                    text_4 = visual.TextStim(win=win, text='+',color='white', height=0.3)
            else:
                cooldown_counter-=1
                #print("cooling down:",cooldown_counter,mainClock.getTime())
                text_4 = visual.TextStim(win=win, text='+',color='white', height=0.3)
        #keep calculating triggers
        else:
            text_4 = visual.TextStim(win=win, text='+',color='white', height=0.3)
        #Drawing cross
        continueRoutine = True
        text_4.draw()
        win.flip()
        
        #Write out information
        with open(filename+'_events.csv', 'a', newline="") as csvfile:
            stim_writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            if frame > BaselineFrames*2:
                if cen_trigger_counter == 1:
                    stim_writer.writerow([onset,exp_tr*7,'trigger_experince_sampling',frame,roi_raw_activations[0],roi_raw_activations[1],pda_timeseries[-1],PDA_moving_average['pda'].iloc[-1],CEN_trigger['pda'].iloc[-1],DMN_trigger['pda'].iloc[-1],cen_trigger_counter,dmn_trigger_counter,CEN_response,CEN_response_rt,CEN_mood_response,CEN_mood_response_rt,CEN_duration_response,CEN_duration_response_rt,0,0,0,0,0,0,0,0])
                    frame_post_trigger=7
                    cen_trigger_counter=0
                elif dmn_trigger_counter == 1:
                    stim_writer.writerow([onset,exp_tr*7,'trigger_experince_sampling',frame,roi_raw_activations[0],roi_raw_activations[1],pda_timeseries[-1],PDA_moving_average['pda'].iloc[-1],CEN_trigger['pda'].iloc[-1],DMN_trigger['pda'].iloc[-1],cen_trigger_counter,dmn_trigger_counter,0,0,0,0,0,0,0,0,0,0,DMN_response,DMN_response_rt,DMN_mood_response,DMN_mood_response_rt,DMN_duration_response,DMN_duration_response_rt])
                    frame_post_trigger=755
                    dmn_trigger_counter=0
                else:
                    if frame_post_trigger == 0:
                        stim_writer.writerow([onset,'1.2','+',frame,roi_raw_activations[0],roi_raw_activations[1],pda_timeseries[-1],PDA_moving_average['pda'].iloc[-1],CEN_trigger['pda'].iloc[-1],DMN_trigger['pda'].iloc[-1],cen_trigger_counter,dmn_trigger_counter,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        frame_post_trigger-=1
                        stim_writer.writerow([onset,'1.2','questions_experince_sampling',frame,roi_raw_activations[0],roi_raw_activations[1],pda_timeseries[-1],PDA_moving_average['pda'].iloc[-1],CEN_trigger['pda'].iloc[-1],DMN_trigger['pda'].iloc[-1],cen_trigger_counter,dmn_trigger_counter,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

                        
            else:
                #onset+=1.2
                stim_writer.writerow([onset,'1.2','trigger_calculation',frame,roi_raw_activations[0],roi_raw_activations[1],pda_timeseries[-1],])
                #print ("direction write:",   roi_write)
        


# convert csv output to BIDS-format tsv
#convert_experiencesampling_csv_to_bids(infile = f'{filename}_roi_outputs.csv')

# display ending text and close window
thank_you_end_run_text.draw()
win.flip()
core.wait(3)

win.close()
core.quit()
