Matlab implementation of the following paper,

"Zhao, et al., Multi-View Clustering via Deep Matrix Factorization, AAAI'17"
http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14647/14497

Abstract: Multi-View Clustering (MVC) has garnered more attention recently since many real-world data are comprised of different representations or views. The key is to explore complementary information to benefit the clustering problem. In this paper, we present a deep matrix factorization framework for MVC, where semi-nonnegative matrix factorization is adopted to learn the hierarchical semantics of multi-view data in a layerwise fashion. To maximize the mutual information from each view, we enforce the non-negative representation of each view in the final layer to be the same. Furthermore, to respect the intrinsic geometric structure in each view data, graph regularizers are introduced to couple the output representation of deep structures. As a non-trivial contribution, we provide the solution based on alternating minimization strategy, followed by a theoretical proof of convergence. The superior experimental results on three face benchmarks show the effectiveness of the proposed deep matrix factorization model.


Acknowledgment:

Partial codes are from the following paper:
George Trigeorgis, Konstantinos Bousmalis, Stefanos Zafeiriou, Bjoern W. Schuller Proceedings of The 31st International Conference on Machine Learning, pp. 1692–1700, 2014 http://jmlr.org/proceedings/papers/v32/trigeorgis14.html

This work is supported in part by the NSF IIS award 1651902, NSF CNS award 1314484, ONR award N00014-12-1-1028, ONR Young Investigator Award N00014-14-1-0484, and U.S. Army Research Office Young Investigator Award W911NF-14-1-0218.




Demo
---------------
Run "demo_DMF" to see the provided example of Yale dataset.



