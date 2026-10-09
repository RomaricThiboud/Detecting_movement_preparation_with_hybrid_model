# Detecting_movement_preparation_with_hybrid_model
This repository stores all scripts to process and analyze MEG data from the corresponding manuscript.

# Abstract
Previous research on brain-computer interfaces (BCIs) has demonstrated the feasibility of decoding different movements. However, less attention has been given to detecting motor preparation (i.e., decoding from brain signals whether or not a participant is about to perform a given movement). In this study, we wanted to assess the performance of a new model, rooted in motor preparation physiology, in detecting an upcoming movement across the period preceding a voluntary action.
Participants (n=18, including 7 females) performed a simple self-paced motor task where they either pressed a button to change an image shown on screen, or let the computer change the image. Participants’ brain activity was recorded using magnetoencephalography and submitted to a newly developed linear ensemble classifier that relies on features extracted using PCA from the temporal, spectral and Riemannian domain. We then compared its performance against other classifiers used in BCIs. 
Our HYbrid Domain Riemannian Ensemble (HYDRE) classifier emerged as the best-performing method, achieving average above-chance decoding up to 1400ms before the action, with a maximum ROC score of 76% at the time of the button press. Moreover, decoding performance in the temporal and Riemannian domains was respectively predictive of Readiness Field and Event-Related Desynchronization magnitudes, two well-known motor preparation markers.
Combining features from the temporal, spectral, and Riemannian domains significantly improved motor preparation decoding compared to single-domain approaches. By leveraging the most discriminative features from each domain, HYDRE yielded consistent performance gains across participants with varying baseline decoding performance, effectively addressing inter-individual variability in motor preparation markers.

# Data
Data are publicly accessible at.... To do

# Scripts

# Funding

