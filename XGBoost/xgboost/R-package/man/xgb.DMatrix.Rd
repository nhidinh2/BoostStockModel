% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.DMatrix.R
\name{xgb.DMatrix}
\alias{xgb.DMatrix}
\alias{xgb.QuantileDMatrix}
\title{Construct xgb.DMatrix object}
\usage{
xgb.DMatrix(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  missing = NA,
  silent = FALSE,
  feature_names = colnames(data),
  feature_types = NULL,
  nthread = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL,
  data_split_mode = "row"
)

xgb.QuantileDMatrix(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  missing = NA,
  feature_names = colnames(data),
  feature_types = NULL,
  nthread = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL,
  ref = NULL,
  max_bin = NULL
)
}
\arguments{
\item{data}{Data from which to create a DMatrix, which can then be used for fitting models or
for getting predictions out of a fitted model.

Supported input types are as follows:
\itemize{
\item \code{matrix} objects, with types \code{numeric}, \code{integer}, or \code{logical}.
\item \code{data.frame} objects, with columns of types \code{numeric}, \code{integer}, \code{logical}, or \code{factor}
}

Note that xgboost uses base-0 encoding for categorical types, hence \code{factor} types (which use base-1
encoding') will be converted inside the function call. Be aware that the encoding used for \code{factor}
types is not kept as part of the model, so in subsequent calls to \code{predict}, it is the user's
responsibility to ensure that factor columns have the same levels as the ones from which the DMatrix
was constructed.

Other column types are not supported.
\itemize{
\item CSR matrices, as class \code{dgRMatrix} from package \code{Matrix}.
\item CSC matrices, as class \code{dgCMatrix} from package \code{Matrix}.
}

These are \strong{not} supported by \code{xgb.QuantileDMatrix}.
\itemize{
\item XGBoost's own binary format for DMatrices, as produced by \code{\link[=xgb.DMatrix.save]{xgb.DMatrix.save()}}.
\item Single-row CSR matrices, as class \code{dsparseVector} from package \code{Matrix}, which is interpreted
as a single row (only when making predictions from a fitted model).
}}

\item{label}{Label of the training data. For classification problems, should be passed encoded as
integers with numeration starting at zero.}

\item{weight}{Weight for each instance.

Note that, for ranking task, weights are per-group.  In ranking task, one weight
is assigned to each group (not each data point). This is because we
only care about the relative ordering of data points within each group,
so it doesn't make sense to assign weights to individual data points.}

\item{base_margin}{Base margin used for boosting from existing model.

In the case of multi-output models, one can also pass multi-dimensional base_margin.}

\item{missing}{A float value to represents missing values in data (not used when creating DMatrix
from text files). It is useful to change when a zero, infinite, or some other
extreme value represents missing values in data.}

\item{silent}{whether to suppress printing an informational message after loading from a file.}

\item{feature_names}{Set names for features. Overrides column names in data frame and matrix.

Note: columns are not referenced by name when calling \code{predict}, so the column order there
must be the same as in the DMatrix construction, regardless of the column names.}

\item{feature_types}{Set types for features.

If \code{data} is a \code{data.frame} and passing \code{feature_types} is not supplied,
feature types will be deduced automatically from the column types.

Otherwise, one can pass a character vector with the same length as number of columns in \code{data},
with the following possible values:
\itemize{
\item "c", which represents categorical columns.
\item "q", which represents numeric columns.
\item "int", which represents integer columns.
\item "i", which represents logical (boolean) columns.
}

Note that, while categorical types are treated differently from the rest for model fitting
purposes, the other types do not influence the generated model, but have effects in other
functionalities such as feature importances.

\strong{Important}: Categorical features, if specified manually through \code{feature_types}, must
be encoded as integers with numeration starting at zero, and the same encoding needs to be
applied when passing data to \code{\link[=predict]{predict()}}. Even if passing \code{factor} types, the encoding will
not be saved, so make sure that \code{factor} columns passed to \code{predict} have the same \code{levels}.}

\item{nthread}{Number of threads used for creating DMatrix.}

\item{group}{Group size for all ranking group.}

\item{qid}{Query ID for data samples, used for ranking.}

\item{label_lower_bound}{Lower bound for survival training.}

\item{label_upper_bound}{Upper bound for survival training.}

\item{feature_weights}{Set feature weights for column sampling.}

\item{data_split_mode}{Not used yet. This parameter is for distributed training, which is not yet available for the R package.}

\item{ref}{The training dataset that provides quantile information, needed when creating
validation/test dataset with \code{\link[=xgb.QuantileDMatrix]{xgb.QuantileDMatrix()}}. Supplying the training DMatrix
as a reference means that the same quantisation applied to the training data is
applied to the validation/test data}

\item{max_bin}{The number of histogram bin, should be consistent with the training parameter
\code{max_bin}.

This is only supported when constructing a QuantileDMatrix.}
}
\value{
An 'xgb.DMatrix' object. If calling \code{xgb.QuantileDMatrix}, it will have additional
subclass \code{xgb.QuantileDMatrix}.
}
\description{
Construct an 'xgb.DMatrix' object from a given data source, which can then be passed to functions
such as \code{\link[=xgb.train]{xgb.train()}} or \code{\link[=predict]{predict()}}.
}
\details{
Function \code{xgb.QuantileDMatrix()} will construct a DMatrix with quantization for the histogram
method already applied to it, which can be used to reduce memory usage (compared to using a
a regular DMatrix first and then creating a quantization out of it) when using the histogram
method (\code{tree_method = "hist"}, which is the default algorithm), but is not usable for the
sorted-indices method (\code{tree_method = "exact"}), nor for the approximate method
(\code{tree_method = "approx"}).

Note that DMatrix objects are not serializable through R functions such as \code{\link[=saveRDS]{saveRDS()}} or \code{\link[=save]{save()}}.
If a DMatrix gets serialized and then de-serialized (for example, when saving data in an R session or caching
chunks in an Rmd file), the resulting object will not be usable anymore and will need to be reconstructed
from the original source of data.
}
\examples{
data(agaricus.train, package = "xgboost")

## Keep the number of threads to 1 for examples
nthread <- 1
data.table::setDTthreads(nthread)
dtrain <- with(
  agaricus.train, xgb.DMatrix(data, label = label, nthread = nthread)
)
fname <- file.path(tempdir(), "xgb.DMatrix.data")
xgb.DMatrix.save(dtrain, fname)
dtrain <- xgb.DMatrix(fname)
}
