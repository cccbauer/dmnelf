# DMNELF Microstate → PDA Decoder Pipeline

## What this project does
Offline EEG → fMRI decoder that predicts the real-time neurofeedback
signal (PDA) from EEG microstates. Replicates and validates the Custo
2017 TESS method for the DMNELF simultaneous EEG-fMRI dataset.
The ultimate goal is to replace the real-time fMRI signal with an
EEG-derived proxy, enabling neurofeedback without an MRI scanner.

## Personnel
- PI: Susan Whitfield-Gabrieli (Sue)
- Researcher: Clemens C.C. Bauer (cccbauer), MD PhD
- Lab: EPIC Brain Lab, Northeastern University
- Affiliate: Gabrieli Lab, MIT McGovern Institute
- Collaborators: Chris (EEG-fMRI pipeline), MGH, Yale, McLean, LMU Munich

## Study: DMNELF
Simultaneous EEG-fMRI neurofeedback study.
7 subjects, 3 tasks (rest, shortrest, feedback), Siemens Prisma 3T.
Target: real-time FPN > DMN activation = ball moves up on screen.
Feedback signal: PDA = CEN minus DMN, delivered via MURFI.

---

## Machines
This project is developed on two local machines:
- cccbauer: /Users/cccbauer/... (primary)
- anitya:   /Users/anitya/...  (secondary, shared machine)

config.py handles both via os.environ USER detection.

---

## Cluster
- Host:    explorer.northeastern.edu
- User:    cccbauer
- Account: suewhit
- Python:  $HOME/my_anaconda/bin/python
- Partition: short (default), sharing (75 idle nodes)
- SSH: passwordless via ~/.ssh/id_ed25519

---

## Cluster Paths