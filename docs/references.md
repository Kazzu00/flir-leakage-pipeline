# References

Only sources already present in the repository documentation are consolidated
here. No bibliography was completed from memory or external searches. Official
implementation documentation is identified as such, not substituted for an
unrecorded original paper citation.

## Models, leakage and reduction literature


- Acosta-Bernal et al.: retained dataset reference; exact title/DOI remains
  unconfirmed in the available documentation.
- Figueiredo & Mendes (2024), [Analyzing Information Leakage on Video Object
  Detection Datasets by Splitting Images Into Clusters With High Spatiotemporal
  Correlation](https://doi.org/10.1109/ACCESS.2024.3383047).
- Radford et al. (2021), [CLIP](https://arxiv.org/abs/2103.00020).
- Oquab et al. (2023), [DINOv2](https://arxiv.org/abs/2304.07193).
- Wang et al. (2021), [PaCMAP and dimensionality-reduction analysis](https://jmlr.org/papers/v22/20-1061.html).
- McInnes, Healy & Astels (2017), [hdbscan](https://joss.theoj.org/papers/10.21105/joss.00205).
- Studer et al. (2021), [CRISP-ML(Q)](https://www.mdpi.com/2504-4990/3/2/20).

## Implementation and visualization sources

- [TSNE de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
- [Trustworthiness de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).
- [PaCMAP oficial](https://github.com/YingfanWang/PaCMAP).
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
- [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html).
- [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).
- [Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
- [scipy.optimize.milp/HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
- [YOLO11, documentación oficial](https://docs.ultralytics.com/models/yolo11/).
- [Configuración de entrenamiento](https://docs.ultralytics.com/modes/train/).
- [Validador fijado a 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/models/yolo/detect/val.py).
- [Definiciones de métricas fijadas a 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/utils/metrics.py).
- [cpietsch/vikus-viewer](https://github.com/cpietsch/vikus-viewer).
- [ccb4e7e3d921521b01b2760b51bf4d1b090b844b](https://github.com/cpietsch/vikus-viewer/tree/ccb4e7e3d921521b01b2760b51bf4d1b090b844b).
- [Licencia MIT](https://github.com/cpietsch/vikus-viewer/blob/ccb4e7e3d921521b01b2760b51bf4d1b090b844b/LICENSE.md).
- [documentación de Streamlit](https://docs.streamlit.io/develop/concepts/architecture/run-your-app).
- [official loading documentation](https://huggingface.co/docs/transformers/models).

## Citation limits

The existing t-SNE, DBSCAN and OPTICS references are implementation documentation.
No original-paper metadata is added here. HDBSCAN has both the retained JOSS
reference and scikit-learn implementation documentation; experiments use the
latter implementation. The precise dataset/class-nomenclature bibliography
remains unresolved; see [class evidence](analysis/dataset_classes.md).
CRISP-ML(Q) is retained as a conceptual reference, not a documentation hierarchy.
