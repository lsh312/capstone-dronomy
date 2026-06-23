## SIFT Google Satellite Anchor Test

Frame used:
- data/drone_test.jpg

Satellite source:
- Google Maps Static API
- Zoom: 20

Results:
- Drone keypoints: 4871
- Satellite keypoints: 13300
- Good matches: 369
- RANSAC inliers: 11
- Inlier ratio: 0.030

Conclusion:
At Google zoom 20, SIFT produced a very rough localization estimate, but the low inlier ratio and poor satellite resolution show that classical SIFT is not robust enough for reliable drone-to-satellite anchoring.