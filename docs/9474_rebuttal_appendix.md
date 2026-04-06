**Table S1:** Grid search results for PI-GNN hyperparameter tuning across different learning rates and dropout probabilities. We report the mean performance ratio and standard deviation ($\bar{\rho} \pm \text{std}$) over 200 training instances (the same instances used to train our evaluator and generator). The optimal configuration (config_id 20) used in our main evaluation is highlighted in bold.

| config_id | learning_rate | dropout | $\bar{\rho}\uparrow\pm\text{std}$ |
| --------- | ------------- | ------- | --------------------------------- |
| 1         | 0.00005       | 0       | 0.7516$\pm$0.2951                 |
| 2         | 0.00005       | 0.1     | 0.8674$\pm$0.1726                 |
| 3         | 0.00005       | 0.2     | 0.8891$\pm$0.1330                 |
| 4         | 0.00005       | 0.3     | 0.9047$\pm$0.0740                 |
| 5         | 0.00005       | 0.5     | 0.9109$\pm$0.0308                 |
| 6         | 0.0001        | 0       | 0.7647$\pm$0.2958                 |
| 7         | 0.0001        | 0.1     | 0.8858$\pm$0.1362                 |
| 8         | 0.0001        | 0.2     | 0.9043$\pm$0.0993                 |
| 9         | 0.0001        | 0.3     | 0.9158$\pm$0.0384                 |
| 10        | 0.0001        | 0.5     | 0.9174$\pm$0.0300                 |
| 11        | 0.0003        | 0       | 0.8513$\pm$0.1883                 |
| 12        | 0.0003        | 0.1     | 0.9146$\pm$0.0411                 |
| 13        | 0.0003        | 0.2     | 0.9216$\pm$0.0428                 |
| 14        | 0.0003        | 0.3     | 0.9229$\pm$0.0401                 |
| 15        | 0.0003        | 0.5     | 0.9249$\pm$0.0315                 |
| 16        | 0.001         | 0       | 0.8943$\pm$0.0683                 |
| 17        | 0.001         | 0.1     | 0.9227$\pm$0.0443                 |
| 18        | 0.001         | 0.2     | 0.9286$\pm$0.0456                 |
| 19        | 0.001         | 0.3     | 0.9291$\pm$0.0454                 |
| 20        | **0.001**     | **0.5** | **0.9318$\pm$0.0338**             |
| 21        | 0.003         | 0       | 0.8925$\pm$0.0717                 |
| 22        | 0.003         | 0.1     | 0.9261$\pm$0.0474                 |
| 23        | 0.003         | 0.2     | 0.9317$\pm$0.0481                 |
| 24        | 0.003         | 0.3     | 0.9316$\pm$0.0484                 |
| 25        | 0.003         | 0.5     | 0.9305$\pm$0.0453                 |



**Table S2:** Performance comparison of our method against the fine-tuned PI-GNN and broader classical heuristics (Simulated Annealing and Goemans-Williamson). Results are reported as the mean performance ratio $\pm$ standard deviation to illustrate the variance across instances.

| Dataset (#) | PI-GNN            | SA                | GW                | Ours              |
| ----------- | ----------------- | ----------------- | ----------------- | ----------------- |
| B (8)       | 0.9416$\pm$0.0157 | 0.8579$\pm$0.0208 | 0.8141$\pm$0.0240 | 0.8417$\pm$0.0256 |
| BE (16)     | 0.9311$\pm$0.0098 | 0.8941$\pm$0.0157 | 0.8664$\pm$0.0209 | 0.8824$\pm$0.0304 |
| W (26)      | 0.9106$\pm$0.0426 | 0.9272$\pm$0.0134 | 0.9473$\pm$0.0117 | 0.9153$\pm$0.0280 |



**Table S3:** Detailed hyperparameter settings for the fine-tuned PI-GNN baseline. These configurations correspond to the optimal setup identified in our grid search and adhere to the recommendations from the official PI-GNN repository.

| parameter      | value  | meaning                                                      |
| -------------- | ------ | ------------------------------------------------------------ |
| learning_rate  | 1e-3   | Adam learning rate; controls optimization step size.         |
| number_epochs  | 100000 | Maximum training epochs per run.                             |
| prob_threshold | 0.5    | Threshold for binarizing node probabilities into partition bits. |
| tolerance      | 1e-4   | Early-stopping loss-delta tolerance.                         |
| patience       | 100    | Early-stopping patience (consecutive epochs).                |
| dropout        | 0.5    | Dropout probability for hidden layers (regularization).      |
| n_layers       | 2      | Number of GNN hidden propagation layers.                     |



**Table S4:** Hyperparameter configuration and annealing schedule for the Simulated Annealing (SA) baseline.

| parameter       | value                                        | meaning                                                |
| --------------- | -------------------------------------------- | ------------------------------------------------------ |
| SA_ITERATIONS   | 150                                          | Number of annealing iterations per run.                |
| initial_temp    | max(1.0, mean_abs_w * max(1, n_v//5))        | Initial temperature controlling early exploration.     |
| final_temp      | 1e-3                                         | Terminal temperature target.                           |
| cooling_rate    | (final_temp/initial_temp)^(1/(iterations-1)) | Geometric cooling factor per iteration.                |
| neighbor_move   | single-bit flip                              | Neighborhood operator: flip one random node each step. |
| acceptance_rule | Metropolis: exp(delta/T) for delta<0         | Probability rule for accepting worse moves.            |



**Table S5:** SDP solver configurations and randomized rounding parameters for the Goemans-Williamson (GW) baseline.

| parameter          | value | meaning                                          |
| ------------------ | ----- | ------------------------------------------------ |
| solver             | SCS   | SDP solver used for GW relaxation.               |
| max_iters          | 25    | Maximum SCS iterations.                          |
| eps                | 1e-4  | SCS convergence tolerance.                       |
| alpha              | 1.8   | SCS relaxation parameter.                        |
| scale              | 5.0   | SCS data scaling parameter.                      |
| random_hyperplanes | 5     | Number of randomized hyperplane rounding trials. |



**Table S6:** Generalization performance across varying problem sizes ($N$) when training on different fractions of the offline dataset (20%, 40%, 60%, 80%, 100%). A statistically significant performance drop compared to the 100% model ($p < 0.05$, one-sided paired t-test) is denoted by (*).

| Train Fraction (Size) | N=51 (p-value)                | N=100 (p-value)               | N=501 (p-value)               | N=800 (p-value)               | N=1000 (p-value)              |
| --------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| 100% (42,069)         | 0.9125 (N/A)                  | 0.9059 (N/A)                  | 0.8935 (N/A)                  | 0.8670 (N/A)                  | 0.7244 (N/A)                  |
| 80% (33,655)          | -                             | -                             | 0.8925 ($7.26\times10^{-2}$)  | 0.8663 ($1.10\times10^{-1}$)  | 0.7229 ($9.60\times10^{-2}$)  |
| 60% (25,241)          | 0.9110 ($2.35\times10^{-1}$)  | 0.9042 ($1.66\times10^{-1}$)  | 0.8923* ($4.41\times10^{-2}$) | 0.8655* ($2.07\times10^{-2}$) | 0.7221* ($2.68\times10^{-2}$) |
| 40% (16,827)          | 0.9087 ($7.12\times10^{-2}$)  | 0.8998* ($4.41\times10^{-2}$) | 0.8919* ($2.16\times10^{-2}$) | 0.8652* ($9.07\times10^{-4}$) | 0.7212* ($3.10\times10^{-3}$) |
| 20% (8,314)           | 0.9030* ($4.35\times10^{-2}$) | 0.9007* ($3.71\times10^{-2}$) | -                             | -                             | -                             |



**Table S7**: Average wall-clock time per fine-tuning step ($\tau$) in milliseconds across different problem sizes ($N$).

| problem sizes ($N$) | $\tau$ (ms) |
| ------------------- | ----------- |
| 51                  | 21          |
| 101                 | 28          |
| 251                 | 59          |
| 501                 | 151         |