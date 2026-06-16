**2026-06-16:**
Right now the feature extraction tool works but is flawed for the following reasons:
- Angle of slouching is not properly calculated. Right now takes the angle of vector between nose and midpoint of left and right shoulders and a horizontal line.
- Features don't capture the position of the laptop itself nor the angle of the screen.

Next steps:
- Find another way of calculating the angle between the neck and the shoulders.
- Maybe assign a weight of when angles matter vs the neck length and shoulder length ratio. 
