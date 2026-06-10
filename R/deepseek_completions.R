# deepseek_completions.R
# DeepSeek Chat Completions API integration for EndpointR
#
# DeepSeek exposes an OpenAI-compatible /chat/completions endpoint, so the
# request-building logic is almost identical to openai_completions.R.
# Key differences:
#   - Base URL  : https://api.deepseek.com/v1/chat/completions
#   - Auth key  : DEEPSEEK_API_KEY  (Bearer token, same header format)
#   - Default model: "deepseek-chat" (maps to V4-Flash; use "deepseek-reasoner" for R1)
#   - Structured outputs: supported via response_format (same schema shape as OpenAI)
#   - Prompt caching: automatic on DeepSeek's side — no extra parameters required.
#     Repeated system prompts / context prefixes are cached automatically.
#   - Batch / off-peak: DeepSeek does not yet offer an async batch endpoint like
#     OpenAI's Batch API. Use concurrent_requests + off-peak scheduling instead.
#
# Model reference (June 2026):
#   "deepseek-chat"      -> DeepSeek V4-Flash  ($0.14 input / $0.28 output per 1M tokens)
#   "deepseek-reasoner"  -> DeepSeek R1        ($0.29 / $0.29 per 1M tokens)
#
# Usage mirrors the oai_* family:
#   ds_complete_text()   - single text
#   ds_complete_chunks() - large vector with chunked parquet output
#   ds_complete_df()     - data-frame interface (wraps ds_complete_chunks)


# ds_build_completions_request ------------------------------------------------

#' Build a DeepSeek Chat Completions request
#'
#' Constructs an httr2 request object for DeepSeek's Chat Completions API.
#' The endpoint is OpenAI-compatible, so the body structure is identical to
#' \code{\link{oai_build_completions_request}}; only the base URL and API key
#' differ.
#'
#' @details
#' DeepSeek automatically caches repeated prompt prefixes (system prompts,
#' document context) on their servers — no extra parameters are required.
#' Cache-hit input tokens are billed at ~2% of the standard rate, which can
#' dramatically reduce costs when the same system prompt is reused across many
#' rows of a dataset.
#'
#' Structured outputs are supported via the same \code{schema} argument used
#' by the OpenAI functions. Pass a \code{json_schema} object created with
#' \code{\link{create_json_schema}}.
#'
#' @param input Text input to send to the model.
#' @param endpointr_id Optional ID that persists through to the response header.
#' @param model DeepSeek model to use. Default \code{"deepseek-chat"} (V4-Flash).
#'   Use \code{"deepseek-reasoner"} for the R1 reasoning model.
#' @param temperature Sampling temperature (0–2). Lower values are more
#'   deterministic. Default 0.
#' @param max_tokens Maximum tokens in the response. Default 500L.
#' @param schema Optional \code{json_schema} object or list for structured output.
#' @param system_prompt Optional system prompt string.
#' @param key_name Name of the environment variable holding the DeepSeek API
#'   key. Default \code{"DEEPSEEK_API_KEY"}.
#' @param endpoint_url DeepSeek API endpoint URL.
#' @param timeout Request timeout in seconds. Default 20.
#' @param max_retries Maximum retry attempts on failure. Default 5.
#'
#' @return An httr2 request object.
#' @export
#' @seealso \href{https://api-docs.deepseek.com/}{DeepSeek API Docs}
ds_build_completions_request <- function(
    input,
    endpointr_id  = NULL,
    model         = "deepseek-chat",
    temperature   = 0,
    max_tokens    = 500L,
    schema        = NULL,
    system_prompt = NULL,
    key_name      = "DEEPSEEK_API_KEY",
    endpoint_url  = "https://api.deepseek.com/v1/chat/completions",
    timeout       = 20,
    max_retries   = 5) {

  stopifnot(
    "input must be a non-empty character string" =
      is.character(input) && length(input) == 1 && nchar(input) > 0,
    "model must be a character string" =
      is.character(model) && length(model) == 1,
    "temperature must be numeric between 0 and 2" =
      is.numeric(temperature) && temperature >= 0 && temperature <= 2,
    "max_tokens must be a positive integer" =
      is.numeric(max_tokens) && max_tokens > 0
  )

  api_key <- get_api_key(key_name)

  # Build messages list (identical structure to OpenAI)
  messages <- list()
  if (!is.null(system_prompt)) {
    if (!is.character(system_prompt) || length(system_prompt) != 1) {
      cli::cli_abort("system_prompt must be a single character string")
    }
    messages <- append(messages,
                       list(list(role = "system", content = system_prompt)))
  }
  messages <- append(messages,
                     list(list(role = "user", content = input)))

  body <- list(
    model       = model,
    messages    = messages,
    temperature = temperature,
    max_tokens  = max_tokens
  )

  # Structured output — DeepSeek accepts the same response_format shape as OpenAI
  if (!is.null(schema)) {
    if (inherits(schema, "EndpointR::json_schema")) {
      schema <- json_dump(schema)
    }
    body$response_format <- schema
  }

  request <- base_request(endpoint_url = endpoint_url,
                          api_key       = api_key) |>
    httr2::req_timeout(timeout) |>
    httr2::req_retry(max_tries      = max_retries,
                     backoff        = ~ 2 ^ .x,
                     retry_on_failure = TRUE) |>
    httr2::req_body_json(body)

  if (!is.null(endpointr_id)) {
    request <- httr2::req_headers(request, endpointr_id = endpointr_id)
  }

  return(request)
}


# ds_build_completions_request_list -------------------------------------------

#' Build a list of DeepSeek requests for concurrent processing
#'
#' Vectorised wrapper around \code{\link{ds_build_completions_request}}.
#' Returns a list of httr2 request objects suitable for
#' \code{\link{perform_requests_with_strategy}}.
#'
#' @param inputs Character vector of text inputs.
#' @param endpointr_ids Optional vector of IDs (same length as \code{inputs}).
#' @inheritParams ds_build_completions_request
#'
#' @return A list of httr2 request objects.
#' @export
ds_build_completions_request_list <- function(
    inputs,
    endpointr_ids = NULL,
    model         = "deepseek-chat",
    temperature   = 0,
    max_tokens    = 500L,
    schema        = NULL,
    system_prompt = NULL,
    max_retries   = 5L,
    timeout       = 30,
    key_name      = "DEEPSEEK_API_KEY",
    endpoint_url  = "https://api.deepseek.com/v1/chat/completions") {

  stopifnot(
    "inputs must be a character vector"       = is.character(inputs),
    "inputs must not be empty"                = length(inputs) > 0,
    "endpointr_ids must match length of inputs" =
      is.null(endpointr_ids) || length(inputs) == length(endpointr_ids)
  )

  invalid_indices <- which(is.na(inputs) | nchar(inputs) == 0)
  if (length(invalid_indices) > 0) {
    cli::cli_abort(
      "Inputs at indices: {invalid_indices} are empty or NA. Filter or amend before proceeding."
    )
  }

  requests <- purrr::map(
    inputs,
    ~ ds_build_completions_request(
      input         = .x,
      model         = model,
      temperature   = temperature,
      max_tokens    = max_tokens,
      schema        = schema,
      system_prompt = system_prompt,
      key_name      = key_name,
      endpoint_url  = endpoint_url,
      max_retries   = max_retries,
      timeout       = timeout
    )
  )

  if (!is.null(endpointr_ids)) {
    requests <- purrr::map2(
      .x = requests,
      .y = endpointr_ids,
      .f = ~ httr2::req_headers(.x, endpointr_id = .y)
    )
  }

  return(requests)
}


# ds_complete_text ------------------------------------------------------------

#' Complete a single text with DeepSeek
#'
#' High-level function that builds a request, performs it, and returns the
#' model's response as a character string (or parsed list when a schema is
#' supplied and \code{tidy = TRUE}).
#'
#' @param text A single non-empty character string.
#' @param model DeepSeek model. Default \code{"deepseek-chat"} (V4-Flash).
#' @param system_prompt Optional system prompt.
#' @param schema Optional \code{json_schema} object for structured output.
#' @param temperature Sampling temperature (0–2). Default 0.
#' @param max_tokens Maximum response tokens. Default 500L.
#' @param key_name Environment variable for the API key. Default
#'   \code{"DEEPSEEK_API_KEY"}.
#' @param endpoint_url DeepSeek endpoint URL.
#' @param max_retries Maximum retries on failure. Default 5L.
#' @param timeout Request timeout in seconds. Default 30.
#' @param tidy If \code{TRUE} and a schema is provided, attempt to parse the
#'   JSON response into a list. Default \code{TRUE}.
#'
#' @return Character string or parsed list.
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple completion
#' ds_complete_text(
#'   text = "Summarise the benefits of prompt caching.",
#'   system_prompt = "You are a helpful assistant. Be concise."
#' )
#'
#' # Structured output
#' sentiment_schema <- create_json_schema(
#'   name   = "sentiment",
#'   schema = schema_object(
#'     sentiment  = schema_string("positive, negative, or neutral"),
#'     confidence = schema_number("score between 0 and 1"),
#'     required   = list("sentiment", "confidence")
#'   )
#' )
#'
#' ds_complete_text(
#'   text          = "I absolutely loved this product!",
#'   system_prompt = "Classify the sentiment.",
#'   schema        = sentiment_schema,
#'   temperature   = 0
#' )
#' }
ds_complete_text <- function(
    text,
    model         = "deepseek-chat",
    system_prompt = NULL,
    schema        = NULL,
    temperature   = 0,
    max_tokens    = 500L,
    key_name      = "DEEPSEEK_API_KEY",
    endpoint_url  = "https://api.deepseek.com/v1/chat/completions",
    max_retries   = 5L,
    timeout       = 30,
    tidy          = TRUE) {

  stopifnot(
    "text must be a single, non-empty character string" =
      is.character(text) && length(text) == 1 && nchar(text) > 0
  )

  req <- ds_build_completions_request(
    input         = text,
    model         = model,
    temperature   = temperature,
    max_tokens    = max_tokens,
    schema        = schema,
    system_prompt = system_prompt,
    key_name      = key_name,
    endpoint_url  = endpoint_url,
    timeout       = timeout,
    max_retries   = max_retries
  )

  tryCatch({
    response <- httr2::req_perform(req)
  }, error = function(e) {
    cli::cli_abort(c(
      "Failed to reach DeepSeek API",
      "i" = "Text: {cli::cli_vec(text, list('vec-trunc' = 50, 'vec-sep' = ''))}",
      "x" = "Error: {conditionMessage(e)}"
    ))
  })

  if (httr2::resp_status(response) != 200) {
    error_msg <- .extract_api_error(response)
    cli::cli_abort(c("DeepSeek API request failed", "x" = "{error_msg}"))
  }

  # DeepSeek returns the same choices[0].message.content path as OpenAI
  content <- .extract_oai_completion_content(response)

  if (!is.null(schema) && tidy && !is.na(content)) {
    content <- tryCatch({
      parsed <- jsonlite::fromJSON(content, simplifyVector = FALSE)
      if (!is.null(schema)) {
        parsed <- validate_response(schema, content)
      }
      parsed
    }, error = function(e) {
      cli::cli_warn(c(
        "Failed to parse structured output from DeepSeek",
        "i" = "Returning raw response",
        "x" = conditionMessage(e)
      ))
      content
    })
  }

  return(content)
}


# ds_complete_chunks ----------------------------------------------------------

#' Process a large text vector through DeepSeek in chunks
#'
#' Divides \code{texts} into chunks of \code{chunk_size}, sends each chunk to
#' the DeepSeek Chat Completions API with up to \code{concurrent_requests}
#' parallel requests, and writes each chunk's results to a \code{.parquet} file
#' in \code{output_dir}. This mirrors \code{\link{oai_complete_chunks}} exactly.
#'
#' @details
#' **Prompt caching**: DeepSeek caches repeated prefixes automatically. If you
#' supply the same \code{system_prompt} across all rows (the common case for
#' classification / extraction tasks), the system prompt tokens are billed at
#' ~2% of the standard rate after the first request. No extra parameters are
#' needed.
#'
#' **Concurrency vs rate limits**: DeepSeek's default rate limit is generous,
#' but start with \code{concurrent_requests = 5} and increase if needed. The
#' function uses exponential back-off retries (\code{max_retries}) to handle
#' transient 429s.
#'
#' **No async batch endpoint**: DeepSeek does not currently offer an
#' OpenAI-style asynchronous Batch API. Use \code{concurrent_requests} and
#' off-peak scheduling (16:30–00:30 UTC historically gives ~50% discounts) for
#' high-volume jobs.
#'
#' @param texts Character vector of texts to process.
#' @param ids Vector of unique identifiers (same length as \code{texts}).
#' @param chunk_size Number of texts per chunk. Default 5000L.
#' @param model DeepSeek model. Default \code{"deepseek-chat"}.
#' @param system_prompt Optional system prompt applied to every request.
#' @param output_dir Directory for \code{.parquet} chunk files. Use
#'   \code{"auto"} (default) for a timestamped directory, or \code{NULL} for a
#'   temp directory.
#' @param schema Optional \code{json_schema} object for structured output.
#' @param concurrent_requests Number of simultaneous requests. Default 5L.
#' @param temperature Sampling temperature (0–2). Default 0.
#' @param max_tokens Maximum tokens per response. Default 500L.
#' @param max_retries Maximum retries per request. Default 5L.
#' @param timeout Request timeout in seconds. Default 30.
#' @param key_name Environment variable for the API key. Default
#'   \code{"DEEPSEEK_API_KEY"}.
#' @param endpoint_url DeepSeek endpoint URL.
#' @param id_col_name Name of the ID column in the output tibble. Default
#'   \code{"id"}. Overridden by \code{ds_complete_df()} to match the source
#'   column name.
#'
#' @return A tibble with columns:
#'   \itemize{
#'     \item \code{<id_col_name>}: Original identifier.
#'     \item \code{content}: Model response (character, or JSON string if schema used).
#'     \item \code{.error}: Logical — \code{TRUE} if the request failed.
#'     \item \code{.error_msg}: Error description, or \code{NA}.
#'     \item \code{.chunk}: Chunk number.
#'   }
#' @export
ds_complete_chunks <- function(
    texts,
    ids,
    chunk_size          = 5000L,
    model               = "deepseek-chat",
    system_prompt       = NULL,
    output_dir          = "auto",
    schema              = NULL,
    concurrent_requests = 5L,
    temperature         = 0,
    max_tokens          = 500L,
    max_retries         = 5L,
    timeout             = 30,
    key_name            = "DEEPSEEK_API_KEY",
    endpoint_url        = "https://api.deepseek.com/v1/chat/completions",
    id_col_name         = "id") {

  stopifnot(
    "texts must be a vector"                           = is.vector(texts),
    "ids must be a vector"                             = is.vector(ids),
    "texts and ids must be the same length"            = length(texts) == length(ids),
    "chunk_size must be a positive integer"            = is.numeric(chunk_size) && chunk_size > 0
  )

  output_dir <- .handle_output_directory(output_dir,
                                         base_dir_name = "ds_completions_batch")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Write metadata for reproducibility / debugging
  metadata <- list(
    endpoint_url        = endpoint_url,
    model               = model,
    chunk_size          = chunk_size,
    n_texts             = length(texts),
    concurrent_requests = concurrent_requests,
    timeout             = timeout,
    max_retries         = max_retries,
    key_name            = key_name,
    timestamp           = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
  jsonlite::write_json(metadata,
                       file.path(output_dir, "metadata.json"),
                       auto_unbox = TRUE, pretty = TRUE)

  # Split into chunks
  n       <- length(texts)
  indices <- seq_len(n)
  chunks  <- split(indices, ceiling(indices / chunk_size))

  all_results <- purrr::imap(chunks, function(chunk_idx, chunk_num) {
    chunk_texts <- texts[chunk_idx]
    chunk_ids   <- ids[chunk_idx]

    cli::cli_alert_info(
      "Processing chunk {chunk_num} of {length(chunks)} ({length(chunk_idx)} texts)..."
    )

    requests <- ds_build_completions_request_list(
      inputs        = chunk_texts,
      model         = model,
      temperature   = temperature,
      max_tokens    = max_tokens,
      schema        = schema,
      system_prompt = system_prompt,
      key_name      = key_name,
      endpoint_url  = endpoint_url,
      max_retries   = max_retries,
      timeout       = timeout
    )

    responses <- perform_requests_with_strategy(
      requests            = requests,
      concurrent_requests = concurrent_requests
    )

    # Tidy function reused from the OpenAI path — same response structure
    tidy_func <- function(resp) {
      content <- .extract_oai_completion_content(resp)
      tibble::tibble(content = content)
    }

    chunk_results <- purrr::imap_dfr(responses, function(resp, i) {
      result <- process_response(resp      = resp,
                                 indices   = chunk_idx[i],
                                 tidy_func = tidy_func)

      # Attach the original ID
      result[[id_col_name]] <- chunk_ids[i]
      result$.chunk          <- as.integer(chunk_num)
      result
    })

    # Write to parquet
    parquet_path <- file.path(output_dir,
                              sprintf("chunk_%04d.parquet", as.integer(chunk_num)))
    arrow::write_parquet(chunk_results, parquet_path)
    cli::cli_alert_success("Chunk {chunk_num} written to {.path {parquet_path}}")

    chunk_results
  })

  results <- dplyr::bind_rows(all_results)

  # Reorder columns to put the ID first, then content, then diagnostics
  id_col  <- id_col_name
  results <- dplyr::select(results,
                           dplyr::all_of(id_col),
                           content,
                           .error,
                           .error_msg,
                           .chunk,
                           dplyr::everything())

  return(results)
}


# ds_complete_df --------------------------------------------------------------

#' Complete a data frame of texts using DeepSeek
#'
#' Data-frame interface to DeepSeek's Chat Completions API. Wraps
#' \code{\link{ds_complete_chunks}} with tidy-select column arguments and
#' preserves the original ID column name in the output.
#'
#' @details
#' Results are written progressively to \code{.parquet} files in
#' \code{output_dir} alongside a \code{metadata.json} file. Add
#' \code{output_dir} to \code{.gitignore} to avoid committing API responses.
#'
#' Failed rows have \code{.error = TRUE} and can be filtered and retried
#' independently.
#'
#' @param df A data frame.
#' @param text_var Unquoted column name containing the input text.
#' @param id_var Unquoted column name for the unique row identifier.
#' @inheritParams ds_complete_chunks
#'
#' @return A tibble with the original ID column and:
#'   \itemize{
#'     \item \code{content}: Model response.
#'     \item \code{.error}: Logical.
#'     \item \code{.error_msg}: Error description or \code{NA}.
#'     \item \code{.chunk}: Chunk number.
#'   }
#' @export
#'
#' @examples
#' \dontrun{
#' reviews <- tibble::tibble(
#'   review_id   = 1:3,
#'   review_text = c(
#'     "Absolutely fantastic!",
#'     "Terrible experience.",
#'     "It was okay, nothing special."
#'   )
#' )
#'
#' # Plain text completions
#' results <- ds_complete_df(
#'   df            = reviews,
#'   text_var      = review_text,
#'   id_var        = review_id,
#'   system_prompt = "Classify the sentiment in one word."
#' )
#'
#' # Structured output with schema
#' schema <- create_json_schema(
#'   name   = "sentiment",
#'   schema = schema_object(
#'     sentiment  = schema_string("positive, negative, or neutral"),
#'     confidence = schema_number("score 0–1"),
#'     required   = list("sentiment", "confidence")
#'   )
#' )
#'
#' results <- ds_complete_df(
#'   df            = reviews,
#'   text_var      = review_text,
#'   id_var        = review_id,
#'   system_prompt = "Classify the sentiment.",
#'   schema        = schema,
#'   temperature   = 0
#' )
#'
#' # Unnest structured results
#' results |>
#'   dplyr::filter(!.error) |>
#'   dplyr::mutate(parsed = purrr::map(content, safely_from_json)) |>
#'   tidyr::unnest_wider(parsed)
#' }
ds_complete_df <- function(
    df,
    text_var,
    id_var,
    model               = "deepseek-chat",
    output_dir          = "auto",
    system_prompt       = NULL,
    schema              = NULL,
    chunk_size          = 1000,
    concurrent_requests = 5L,
    max_retries         = 5L,
    timeout             = 30,
    temperature         = 0,
    max_tokens          = 500L,
    key_name            = "DEEPSEEK_API_KEY",
    endpoint_url        = "https://api.deepseek.com/v1/chat/completions") {

  text_sym <- rlang::ensym(text_var)
  id_sym   <- rlang::ensym(id_var)

  stopifnot(
    "df must be a data frame"          = is.data.frame(df),
    "df must not be empty"             = nrow(df) > 0,
    "text_var must exist in df"        = rlang::as_name(text_sym) %in% names(df),
    "id_var must exist in df"          = rlang::as_name(id_sym) %in% names(df),
    "model must be a character string" = is.character(model),
    "chunk_size must be a positive integer" =
      is.numeric(chunk_size) && chunk_size > 0
  )

  output_dir  <- .handle_output_directory(output_dir, base_dir_name = "ds_completions_batch")
  text_vec    <- dplyr::pull(df, !!text_sym)
  id_vec      <- dplyr::pull(df, !!id_sym)
  id_col_name <- rlang::as_name(id_sym)

  ds_complete_chunks(
    texts               = text_vec,
    ids                 = id_vec,
    model               = model,
    system_prompt       = system_prompt,
    schema              = schema,
    chunk_size          = chunk_size,
    concurrent_requests = concurrent_requests,
    max_retries         = max_retries,
    timeout             = timeout,
    temperature         = temperature,
    max_tokens          = max_tokens,
    key_name            = key_name,
    endpoint_url        = endpoint_url,
    output_dir          = output_dir,
    id_col_name         = id_col_name
  )
}