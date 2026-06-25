**2026-06-16:**
Right now the feature extraction tool works but is flawed for the following reasons:
- Angle of slouching is not properly calculated. Right now takes the angle of vector between nose and midpoint of left and right shoulders and a horizontal line.
- Features don't capture the position of the laptop itself nor the angle of the screen.

Next steps:
- Find another way of calculating the angle between the neck and the shoulders.
- Maybe assign a weight of when angles matter vs the neck length and shoulder length ratio.

**2026-06-23:**
Seem to able to normalize neck length when facing different directions. However, Mediapipe seems to not be able to give accurate shoulder width measures when not facing the camera.
I believe this is due to inaccurate depth measures.

Next steps:
- Try out other depth estimation models, e.g depth-pro by apple
