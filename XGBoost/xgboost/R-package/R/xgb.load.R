#' Load XGBoost model from binary file
#'
#' Load XGBoost model from binary model file.
#'
#' @param modelfile The name of the binary input file.
#'
#' @details
#' The input file is expected to contain a model saved in an XGBoost model format
#' using either [xgb.save()] in R, or using some
#' appropriate methods from other XGBoost interfaces. E.g., a model trained in Python and
#' saved from there in XGBoost format, could be loaded from R.
#'
#' Note: a model saved as an R object has to be loaded using corresponding R-methods,
#' not by [xgb.load()].
#'
#' @return
#' An object of `xgb.Booster` class.
#'
#' @seealso [xgb.save()]
#'
#' @examples
#' \dontshow{RhpcBLASctl::omp_set_num_threads(1)}
#' data(agaricus.train, package = "xgboost")
#' data(agaricus.test, package = "xgboost")
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#'
#' train <- agaricus.train
#' test <- agaricus.test
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   nrounds = 2,
#'   params = xgb.params(
#'     max_depth = 2,
#'     eta = 1,
#'     nthread = nthread,
#'     objective = "binary:logistic"
#'   )
#' )
#'
#' fname <- file.path(tempdir(), "xgb.ubj")
#' xgb.save(bst, fname)
#' bst <- xgb.load(fname)
#' @export
xgb.load <- function(modelfile) {
  if (is.null(modelfile))
    stop("xgb.load: modelfile cannot be NULL")

  bst <- xgb.Booster(
    params = list(),
    cachelist = list(),
    modelfile = modelfile
  )
  bst <- bst$bst
  # re-use modelfile if it is raw so we do not need to serialize
  if (typeof(modelfile) == "raw") {
    warning(
      paste(
        "The support for loading raw booster with `xgb.load` will be ",
        "discontinued in upcoming release. Use `xgb.load.raw` instead. "
      )
    )
  }
  return(bst)
}
