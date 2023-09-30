# rl-state-tester
A state tester using rlviser_py that helps you deal with reward tuning or state efficiency, it also works without rlviser, but you'll have a hard time debugging and it was mainly thought to work with it, i'll try to maintain it working without rlviser, maybe it'll break.

**Code is still quite a mess, consider yourself warned**

## Features
- Multiple environment harvesters such as states or rewards that allows you to store a certain amount of these and study whatever you need to
- Multiple state harvesters such as player position/velocity and ball position/velocity, allowing you to extract a given data in a state

You can create your own harvesters, it should work (i said "should").

## How to use
If you want to render the environment, you'll need [rlviser](https://github.com/VirxEC/rlviser), follow the readme in this project, and move the .exe file of both rlviser and umodel in the root folder along with the assets.path pointing to the right path
