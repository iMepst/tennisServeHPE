# Angle Definitions for the Rule-Based Serve Evaluation

Internal reference. The operational definitions below are fixed for the
implementation and are intended to feed, largely verbatim, into the methodology
chapter. Landmark indices refer to the 33-point BlazePose topology.

## Recording convention and coordinate choice

Serves are recorded from a lateral viewpoint with the serving arm facing the
camera. The image plane therefore approximates the player's sagittal plane.

BlazePose returns two coordinate systems per landmark from the same monocular
frame. The image landmarks give in-plane coordinates in the image together with
a relative depth value. The world landmarks give metric coordinates in meters
with the origin at the hip center and are free of camera projection. Both are
model estimates from a single view. The world landmarks are inferred by the
underlying body model rather than measured, and their depth component is the
least reliable part of the output, because depth from a single camera is
mathematically underdetermined.

All angles below are computed in the two-dimensional image plane from the
projected in-plane coordinates. This is a deliberate decision between two
competing error sources rather than an assumption of convenience.

A planar angle is exact only when the relevant segment lies parallel to the
image plane. Any rotation of the segment out of that plane shortens its
projection and biases the measured angle. This projection error is the cost of
the 2D definition. The alternative, computing angles from the world landmarks,
avoids projection error in principle but inherits the depth-estimation noise,
which is largest during the fast and self-occluded phases around ball contact.
The 2D definition accepts the projection error in order to avoid the depth
noise, on material where the two rule angles remain close to the image plane
under the prescribed lateral viewpoint.

The validation design supports this choice. The human reference rater judges the
same monocular video, so a 2D system measures the quantity the human reference
actually assesses. The opposing consideration is stated openly. The reference
values of Jacquier-Bret and Gorce (2024) derive from three-dimensional marker or
multi-camera studies, so comparing a 2D projected angle against a 3D-derived
reference introduces a definition mismatch. This mismatch is reported as a
threat to validity rather than concealed.

The projection error is not uniform across the two rules. Trunk inclination at
the trophy position is predominantly a sagittal motion and stays near the image
plane, so its projection error is small. Elbow flexion at ball impact is the
adverse case, because shoulder internal rotation and the cartwheel action rotate
the hitting arm out of the sagittal plane at exactly the instant the angle is
read. There the 2D foreshortening is largest and the world-landmark depth is
weakest at the same time.

## Empirical control: 2D versus 3D angle

Stage 1 persists both the image coordinates and the world coordinates, so the
choice above can be checked against evidence instead of assumed. On the
validation clips, each rule angle is computed twice, once from the 2D image
projection and once from the 3D world landmarks, at the same event frame. The
agreement is reported per angle and event as the mean and maximum absolute
difference in degrees. Where the two definitions diverge substantially, in
particular for elbow flexion at impact, the divergence is treated as a finding
and the primary definition is re-examined against it. This comparison belongs to
Stage 2 and is recorded here so the coordinate choice is documented as an
examined decision, not an unexamined default.

## Landmark reference

- Shoulders: 11 (left), 12 (right)
- Elbows: 13 (left), 14 (right)
- Wrists: 15 (left), 16 (right)
- Hips: 23 (left), 24 (right)

The serving-side triplet (shoulder, elbow, wrist) is selected according to the
player's dominant arm. Shoulder and hip midpoints are the arithmetic mean of the
left and right landmark coordinates.

## Rule 1: Trunk inclination at trophy position

Reference value: 25.0 +/- 7.1 degrees (Jacquier-Bret and Gorce, 2024).

The trunk is represented by the vector from the hip midpoint (mean of 23 and 24)
to the shoulder midpoint (mean of 11 and 12). Trunk inclination is the angle of
this vector against the image vertical. An upright trunk yields 0 degrees.

This is an absolute inclination measured against a fixed reference (gravity
direction in the image), corresponding to the "upper torso position against the
absolute reference" convention. It is not the pelvis-relative trunk tilt. The
absolute definition is chosen because the pelvis-relative variant requires a
stable pelvic axis, which the lateral viewpoint cannot provide. The two hip
landmarks project onto nearly the same image location and their connecting axis
becomes unreliable.

Limitation to document. The lateral viewpoint captures the sagittal
(forward and backward) component of trunk inclination and does not capture the
lateral (frontal-plane) component. Because this component stays close to the
image plane at the trophy position, the projection error of the 2D definition is
small for this rule. The reference value aggregates studies with mixed camera
conventions and angle definitions, which partly explains its wide dispersion.
The chosen definition is therefore declared transparently and its deviation from
some source conventions is stated as a limitation rather than concealed.

## Rule 2: Elbow flexion at ball impact

Reference value: 30.1 +/- 15.9 degrees (Jacquier-Bret and Gorce, 2024).

Elbow flexion follows the ISB convention, under which a fully extended arm
corresponds to 0 degrees of flexion. It is computed as 180 degrees minus the
geometric angle at the elbow formed by the shoulder, elbow, and wrist landmarks
of the serving arm (11-13-15 or 12-14-16). A value of 30.1 degrees therefore
describes an arm that is close to, but not fully, extended at contact.

This conversion is required because several source studies reported the direct
inter-segment angle, which yields 180 degrees for aligned segments. The
reference value is ISB-homogenized, so the implementation must produce the
ISB-conformant flexion angle to remain comparable.

Limitation to document. This is the angle most exposed to the projection error
of the 2D definition. At ball impact the hitting arm is rotated out of the
sagittal plane, so its image projection is foreshortened and the measured
flexion is biased. The bias direction is systematic and is examined through the
2D versus 3D control above. This rule is therefore the primary case for which
the coordinate choice must be justified with evidence rather than assumption.

## Rule 3 (optional): Front knee flexion at trophy position

Reference value: 64.5 +/- 9.7 degrees (Jacquier-Bret and Gorce, 2024).

Included only if the front knee (hip, knee, ankle of the front leg) is reliably
visible in the lateral recording. Retained as optional pending confirmation of
visibility across the amateur footage. If included, it follows the same ISB
convention as the elbow angle, with an extended leg corresponding to 0 degrees.

## Event definitions (reference, to be fixed separately)

- Trophy position: first peak of vertical racket displacement, coinciding with
  the lowest vertical elbow position and maximum knee flexion.
- Ball impact: instant of contact between racket and ball.

Landmark-based proxies for these events (for example, peak wrist height for
trophy position) are an open implementation decision and are not fixed here.
