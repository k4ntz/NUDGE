# Getout

Getout is a platformer environment inspired by CoinRun.

# Game
__Screen__  
Top left: rewards (higher is better, avoid negative rewards)  
Top right current score (higher is better)

__Game play__  
The level starts once the first action is taken.  
Finish each level with a high score.  
The blue enemy can only be killed by collecting the diamond powerup first!

__Keys__  
W/SPACE: Jump  
A,S: Move left/right  
R: Reset level

# Requirements
```
python >= 3.4 (developed on 3.8)
numpy
Pillow
pyglet
threading
```

# Data recording
```python ./getout_recorder.py```
Each level is saved into the `recordings` directory once the level terminates (by reaching the flag or the score dropping to zero).
