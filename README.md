Motivation & Data Understanding
The core objective of this project is to create an AI-powered system that can take an image of an
outfit and identify the article clothing and automatically estimate its market price, enabling
consumers to make faster and more informed fashion purchasing decisions. The system first
identifies the type of clothing present, ranging from coats to t-shirts to ankle boots (Phase 1),
then predicts price based on learned relationships between visual features and market value using
a structured H&M product dataset (Phase 2). This supports a real-world business use case where
visual appearance strongly correlates with pricing in retail fashion markets with limitless
applications as well, such as automated resale pricing, inventory valuation, and visual
recommendation search. These applications eliminate manual appraisal for online resale markets,
scale visual auditing of large product catalogs, and enable budget-guided browsing and outfit
styling tools. Leveraging machine learning practices, this project produces a “Shazam-style”
outfit recognition and cost calculator for consumers. These applications can also be utilized or
incorporated into existing fashion websites and companies such as Depop, StockX, Instagram,
etc.

Data Preparation
To train a robust model capable of both classification and price regression, two distinct datasets
were utilized, each tailored to a specific phase of the project pipeline.
The industry-standard FashionMNIST dataset served as the primary benchmark for validating
the neural network architecture. The collection comprises 70,000 grayscale images across 10
distinct clothing categories, partitioned into 60,000 training samples and 10,000 testing samples.
While originally 28x28 grayscale, we adapted the dataset into 3-channel RGB and resized them
into 224x224 resolution to match the input format required by ImageNet-pretrained ResNet
models. The processing involved converting images into PyTorch tensors and normalizing pixel
values to a range of [-1, 1] using a mean and standard deviation of 0.5. This standardization is
crucial for stabilizing the gradient descent during training. To maximize computational
efficiency, the data loading utilized a batch size of 128 and shuffling was enabled for the training
set to prevent order bias, ensuring the network learned generalized visual features rather than
memorizing sequence patterns.
Following architectural validation, the H&M Product Dataset was integrated to map visual
aesthetics to their corresponding market value. Ground-truth price labels were derived from the
metadata file, which provides structured attributes such as product type, description, department
classification, and more.
In contrast to the lower-resolution FashionMNIST benchmarks, this dataset utilizes
high-resolution product imagery. The higher fidelity allows the regression model to detect
fine-grained details such as fabric texture, stitching complexity, and material quality that are
critical for distinguishing high-value items from basic staples. This visual richness is necessary
for the model to capture the economic signals embedded in fashion imagery

Modeling
Our modeling strategy follows a two-stage deep learning architecture: feature extraction & price
estimation. We evaluated simple neural network architectures first, but accuracy plateaued
(95.43%) from the FashionMNIST test dataset. To capture richer image features, we upgraded to
a RestNet-18 architecture modified specifically for the FashionMNIST dataset. This removed
early max-pooling to preserve unique details on inputted 28x28 images and customized the FC
layer with 10 outputs that maps to the FashionMNIST class labels. These customizations
significantly improved discriminatory power between categories, such as coats vs. pullovers and
shoes vs. sandals.
The reasoning behind leveraging ImageNet is due to its massive and diverse training across over
a million images of 1,000 categories. Consequently, the model has learned to recognize complex
textures, edges, shapes, lighting variations, patterns, and even pose structures. These low-level
and mid-level visual features transfer extremely well to fashion tasks. This makes ImageNet-pre
trained architectures, such as ResNet18 or EfficientNet, ideal for fine-tuning on FashionMNIST
as we only need to teach it the new labels.
The next stage of our architecture focuses on translating visual understanding into numerical
market value predictions. After the ResNet-18 model processes an image and learns its visual
representation, we remove the final softmax classification layer and replace it with a regression
head, designed to output a continuous price estimate. Rather than retraining the entire network,
we implemented a transfer learning approach as the earlier convolutional layers are frozen,
preserving their learned weights. The last layers of the network are retrained on price data,
allowing the model to learn statistical correlations between visual cues and retail pricing.
Traditionally, classification models are optimized using cross-entropy loss, where predictions are
treated discretely. Price prediction, however, is continuous forecasting, so we changed the
objective function to Root Mean Squared Error (RMSE), directly measuring how close the
predicted value is to its real price. For each FashionMNIST category, the system identifies
matching products in the H&M data using keyword matching. This leads to a median calculation
of matched products and maps the classifier output to the corresponding price range, concluding
in an estimated value. This shift enables the network to learn fine-grained price distinctions
driven by subtle aesthetic factors, such as stitching, garment layering, or labeled luxury.
By grounding price estimation in rich visual feature embeddings rather than simple categorical
encodings, the model becomes capable of capturing the underlying signal of fashion imagery.
This ensures that the output is not merely the mean price of a category, but rather an intentional
valuation based on how premium the product looks, setting the stage for realistic and
consumer-valuable price predictions.

Implementation
To effectively operationalize our strategy, we implemented a training pipeline optimized for both
speed and accuracy using PyTorch, leveraging GPU acceleration to efficiently process large
image batches. We trained the network using a batch size of 128. To combat overfitting and
strengthen model generalization, we introduced lightweight data augmentation, specifically
horizontal flipping and random cropping. This exposed the network to realistic variations in item
alignment and presentation without distorting important visual features. The AdamW optimizer
was selected due to its improved weight-decay performance over the standard Adam optimizer,
helping the model retain strong generalization when learning complex visual representations.
Additionally, a cosine-annealing learning rate scheduler was used to gradually reduce learning
intensity throughout each epoch, ensuring smoother convergence of parameters and avoiding
premature stagnation. Model checkpoints were continuously tracked, and the best-performing
version was preserved for transfer learning into the price regression stage.
By combining these techniques, our implementation ensured rapid learning in early epochs,
stability in later epochs, and higher confidence that the trained network would serve as a robust
feature extractor for downstream price prediction.

Results & Evaluation (See Appendix for Visualizations & Metrics)
The results from our Phase 1 classification training demonstrate strong model competency in
visually distinguishing among fashion categories. Within the first three epochs, the ResNet-18
architecture surpassed the 90% test accuracy threshold, showing that the network immediately
recognized data structure and visual patterns. At convergence, the model achieved a 95.43%
accuracy on the FashionMNIST test set, confirming that the architectural adjustments, such as
the removal of max pooling and the adaptation for grayscale imagery, were beneficial for
extracting fine-grained clothing details. This performance aligns well with the objective of
building a highly reliable visual understanding foundation before transitioning into price
estimation. Our system also outperformed a simpler CNN architecture by a few percentage
points, and offers stronger generalization with fewer training epochs due to pre-trained ImageNet
features. This ensures that the visual representation learned in Phase 1 provides a high-quality
foundation for price prediction.
One important setback, however, is that this approach assigns the same price to all items in one
category, coats, at $64.55. This happens because the model is currently ignoring factors such as
material quality and brand signals, since we have not trained it on a sufficiently large or diverse
dataset. This could be solved with a significantly larger dataset or more computing power than
Colab Pro could handle.
In parallel, we conducted a POC regression benchmark using H&M product metadata to validate
whether price signals could be effectively learned. The model achieved an RMSE of
approximately $10.96 and an R2 score of 0.75, demonstrating that meaningful price relationships
already exist in the structured product attributes alone. Once visual embeddings from our trained
classifier are integrated into this regression pipeline, we expect improved price estimation
accuracy, especially for garments with distinct luxury or craftsmanship characteristics. All
together, our Phase 1 evaluation confirms that the model not only sees fashion items clearly but
can also begin to infer pricing logic: a key step toward successful competition of our business
goal.
Even at this early stage, the results show that the platform is on track for reliable commercial use
in several high-value applications. WIth high classification accuracy, price estimation accuracy,
real-time identification, and personalization, market platforms could easily integrate this model
to estimate fair value of uploaded listings. Ecommerce retailers can also use the classifier to
continuously track perceived product value, stabilize price transparency, and inform pricing and
trend forecasting.
In short, our model is simple and interpretable with no additional model training required. It is
also incredibly robust to outliers since we leverage median values over mean with fast inference.
It is, however, limited by visual nuances (brand, material, condition), ten broad clothing
categories, and cannot capture premium features. With more data to ingest, our results indicate a
clear path to monetization and can unlock operational efficiencies for fashion marketplaces. To
support a competitive business strategy.

Deployment
Successful deployment of this system opens the door to powerful real-world applications,
particularly in consumer shopping experiences and retail analytics. The primary deployment
vision is a mobile or web-based interface where users can upload or capture a photo of an outfit
and instantly receive both a garment classification and an estimated valuation. This enables
fashion discovery instantaneously in the moment: on big city streets, in stores, or across social
media content. On the business end, the system could integrate with retail or resale platforms,
enhancing product searchability and driving conversion by recommending similar options across
different price tiers. The platform could generate revenue through affiliate commerce links or
curated marketplace listings aligned with user budget preferences. Anonymized behavioral data
from users could also help retailers with trend forecasting, inventory optimization, and real-time
market positioning. These benefits reinforce the value proposition initially described in our
proposal and emphasize the commercialization potential of an AI-driven price-perception model
in the fashion field.
Some critical considerations to account for include the quality of real-world imagery, which can
vary in lighting, background clutter, pose, or occlusion. To prevent degraded performance,
deployment would require continuous retraining on high-resolution fashion images of diverse
sources, ideally from the retailer’s own catalog. The valuation accuracy must also scale beyond
the category medians. Incorrect pricing predictions could show heavy churn rates, distort the
customer experience, and even expose the business to liability concerns if suggest prices are
misaligned with market expectations.
Finally, ethical issues must be transparently handled and stated with consumers. Uploaded
images must be handled securely with strong encryption, including prompt deletion policies. The
model must also be continuously tested for bias, such as associating higher prices with certain
demographics, styling cues, or body types.

Conclusion
Overall, this project demonstrates a strong technical foundation and a clear pathway toward
business-viable innovation. We began by establishing a robust fashion-image understanding
system capable of reliably identifying core clothing categories with over 94% accuracy. That
success directly validates our architectural approach and enables a seamless transition into Phase
2: price regression using visual feature representations. Early benchmark pricing results also
indicate that economic signals in fashion can be learned algorithmically, even before
incorporating visual embeddings from our trained classifier. The full system, once deployed, can
offer users immediate insight into style classification and predicted value, bridging the gap
between visual inspiration and AI-intelligent fashion guidance (Phase 3). With some minor
setbacks, more data and computing power will be required to properly scale this product to its
most effective level, including more fashion categories, material recognition, updated social
sentiment, etc. As we move forward, integrating H&M image data and refining our regression
head will transform the model from a high-accuracy classifier into a real-time fashion valuation
tool, fulfilling the precise vision outlined in our proposal and expanding the future possibilities
of AI retail experiences.
