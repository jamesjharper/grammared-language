"""
gRPC server implementation compatible with LanguageTool ML Server protocol (ml_server.proto).

Implements ProcessingServer and MLServer services.
"""

import logging
from concurrent import futures
import grpc
import time
import sys
import os
from threading import Condition, Thread, Lock
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add parent directory to path for absolute imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from grammared_language.clients.async_multi_client import AsyncMultiClient
from grammared_language.clients.gector_client import GectorClient
from grammared_language.clients.coedit_client import CoEditClient
# from grammared_language.utils.grammar_correction_extractor import GrammarCorrectionExtractor
# from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor
from grammared_language.api.util import SimpleCacheStore
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.api.grpc_gen import ml_server_pb2, ml_server_pb2_grpc
from grammared_language.utils.config_parser import create_clients_from_config, get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for clients
correction_multi_client = None
client_lock = Lock()
client_condition = Condition(client_lock)
_client_inflight = {}
_model_generation = 0

def initialize_clients(max_retries: int = 5, retry_delay: int = 10):
    """
    Initialize clients from config with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        AsyncMultiClient instance
        
    Raises:
        Exception: If initialization fails after all retries
    """
    config_path = os.getenv('GRAMMARED_LANGUAGE__MODEL_CONFIG_PATH', 'model_config.yaml')
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Initializing clients (attempt {attempt}/{max_retries})...")
            clients = create_clients_from_config(config_path=config_path)
            logger.info(f"Successfully loaded {len(clients)} client(s)")
            
            multi_client = AsyncMultiClient(clients=clients)
            logger.info("Client initialization successful")
            
            # Warm-up prediction to ensure models are loaded
            logger.info("Running warm-up prediction...")
            warmup_text = "these is test sentence for warm up model."
            multi_client.predict_with_merge(warmup_text)
            logger.info("Warm-up prediction completed")
            
            return multi_client
            
        except Exception as e:
            logger.error(f"Failed to initialize clients (attempt {attempt}/{max_retries}): {str(e)}", exc_info=True)
            
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.critical("All client initialization attempts failed")
                raise

def invalidate_model_result_caches():
    """Discard grammar results produced by the previously active client."""
    process_cache_store.clear()
    match_cache_store.clear()
    match_anylized_cache_store.clear()


def _acquire_client():
    """Return a client lease that remains valid until ``_release_client``."""
    with client_condition:
        if correction_multi_client is None:
            raise RuntimeError("Grammar clients have not been initialized")
        client = correction_multi_client
        client_id = id(client)
        _client_inflight[client_id] = _client_inflight.get(client_id, 0) + 1
        return client, _model_generation


def _release_client(client):
    with client_condition:
        client_id = id(client)
        remaining = _client_inflight[client_id] - 1
        if remaining:
            _client_inflight[client_id] = remaining
        else:
            del _client_inflight[client_id]
            client_condition.notify_all()


def _retire_client(client):
    """Close an old client only after requests holding its lease have finished."""
    if client is None:
        return

    with client_condition:
        while _client_inflight.get(id(client), 0):
            client_condition.wait()

    close = getattr(client, "close", None)
    if close:
        close()


def activate_client(new_client):
    """Atomically make a warmed client current and retire its predecessor."""
    global correction_multi_client, _model_generation

    with client_condition:
        old_client = correction_multi_client
        correction_multi_client = new_client
        _model_generation += 1
        logger.info("Clients reloaded successfully")

    # Generation-qualified keys prevent an old in-flight request from making
    # its result visible to requests using the newly activated client.
    invalidate_model_result_caches()
    _retire_client(old_client)


def reload_clients_periodically(reload_interval: int = 3600):
    """
    Periodically reload clients in a background thread.
    
    Args:
        reload_interval: Interval in seconds between reloads (default: 3600 = 1 hour)
    """
    while True:
        try:
            time.sleep(reload_interval)
            logger.info("Starting periodic client reload...")
            
            new_client = initialize_clients()
            
            activate_client(new_client)
                
        except Exception as e:
            logger.error(f"Failed to reload clients: {str(e)}", exc_info=True)
            logger.info("Continuing with existing clients")

# # Initialize ERRANT Grammar Correction Extractor
# errant_extractor = ErrantGrammarCorrectionExtractor(language='en', min_length=1)

cache_size = int(os.getenv('GRAMMARED_LANGUAGE__GRPC_SERVER_CACHE_SIZE', '100000'))
analyze_cache_store = SimpleCacheStore(cache_size)
process_cache_store = SimpleCacheStore(cache_size)
match_cache_store = SimpleCacheStore(cache_size)
match_anylized_cache_store = SimpleCacheStore(cache_size)


def _model_cache_key(generation, text):
    return generation, text


def initialize_runtime():
    """Initialize the first client and start periodic reloads exactly once."""
    global correction_multi_client
    with client_lock:
        already_initialized = correction_multi_client is not None
    if already_initialized:
        return

    new_client = initialize_clients()
    activate_client(new_client)
    reload_interval = int(os.getenv('GRAMMARED_LANGUAGE__CLIENT_RELOAD_INTERVAL', '3600'))
    reload_thread = Thread(
        target=reload_clients_periodically, args=(reload_interval,), daemon=True
    )
    reload_thread.start()
    logger.info(f"Started client reload thread (interval: {reload_interval}s)")

# Health check state
service_healthy = False

def pydantic_match_to_ml_match(match, offset_adjustment: int = 0) -> ml_server_pb2.Match:
    """Convert Pydantic Match model to ml_server Match."""
    grpc_match = ml_server_pb2.Match(
        offset=match.offset + offset_adjustment,
        length=match.length,
        id=match.id or "GrammaredLanguage",
        sub_id=match.sub_id or "",
        suggestions=match.suggestions or [],
        ruleDescription=match.rule.description if match.rule else None,
        matchDescription=match.message or "",
        matchShortDescription=match.shortMessage or match.message or "",
        url=match.url or "",
        suggestedReplacements=[
            ml_server_pb2.SuggestedReplacement(
                replacement=r.replacement,
                description=r.description or "",
                suffix=r.suffix or "",
                confidence=r.confidence if r.confidence is not None else -1,
            )
            for r in (match.suggested_replacements or [])
        ],
        autoCorrect=True,
        type=ml_server_pb2.Match.MatchType.UnknownWord,  # Grammar errors are "Other" type
        contextForSureMatch=0,
        rule=ml_server_pb2.Rule(
            sourceFile=match.rule.id or "grammared_language",
            issueType=match.rule.issueType or "style",
            tempOff=False,
            category=ml_server_pb2.RuleCategory(
                id=match.rule.id or "grammared_language",
                name=match.rule.description or "Style Suggestion"
            ) if match.rule.category else None,
            isPremium=False,
            tags=[]
        ) if match.rule else None
    )

    return grpc_match


class HealthCheckHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Docker health checks."""
    
    def do_GET(self):
        """Handle GET requests for health check."""
        if self.path == '/health' or self.path == '/healthz':
            if service_healthy:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'OK')
            else:
                self.send_response(503)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Service not ready')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress HTTP server logging."""
        pass


class ProcessingServerServicer(ml_server_pb2_grpc.ProcessingServerServicer):
    """gRPC servicer for LanguageTool ProcessingServer."""

    def Analyze(self, request: ml_server_pb2.AnalyzeRequest, context: grpc.ServicerContext) -> ml_server_pb2.AnalyzeResponse:
        """
        Analyze text and return analyzed sentences (tokenization, POS tagging, etc).
        
        Args:
            request: AnalyzeRequest containing text and processing options
            context: gRPC context
            
        Returns:
            AnalyzeResponse with analyzed sentences
        """
        try:
            logger.info(f"Analyze request: language={request.options.language}, text_len={len(request.text)}")
            
            # Simple tokenization - just split by spaces for now
            # In production, use a proper NLP library
            text = request.text
            if not text:
                return ml_server_pb2.AnalyzeResponse(sentences=[])

            if analyze_cache_store.contains(text):
                analyzed_sentences = analyze_cache_store.get(text)
                return ml_server_pb2.AnalyzeResponse(sentences=analyzed_sentences)
            
            sentences = text.split('. ')
            analyzed_sentences = []
            
            for sentence in sentences:
                tokens = sentence.split()
                token_readings = []
                pos = 0
                
                for token in tokens:
                    token_reading = ml_server_pb2.AnalyzedTokenReadings(
                        readings=[
                            ml_server_pb2.AnalyzedToken(
                                token=token,
                                posTag="NN",  # Default POS tag
                                lemma=token.lower()
                            )
                        ],
                        chunkTags=[],
                        startPos=pos
                    )
                    token_readings.append(token_reading)
                    pos += len(token) + 1
                
                analyzed_sentences.append(
                    ml_server_pb2.AnalyzedSentence(
                        text=sentence,
                        tokens=token_readings
                    )
                )
            
            analyze_cache_store.add(text, analyzed_sentences)
            return ml_server_pb2.AnalyzeResponse(sentences=analyzed_sentences)
            
        except Exception as e:
            logger.error(f"Error in Analyze RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise

    def Process(self, request: ml_server_pb2.ProcessRequest, context: grpc.ServicerContext) -> ml_server_pb2.ProcessResponse:
        """
        Process analyzed sentences and return grammar matches.
        
        Args:
            request: ProcessRequest containing analyzed sentences and options
            context: gRPC context
            
        Returns:
            ProcessResponse with grammar error matches
        """
        try:
            logger.info(f"Process request: {len(request.sentences)} sentences, language={request.options.language}")
            
            client, generation = _acquire_client()
            try:
                results = [None] * len(request.sentences)
                missing_indices = {}
                missing_texts = []
                for index, sentence in enumerate(request.sentences):
                    text = sentence.text
                    cache_key = _model_cache_key(generation, text)
                    if process_cache_store.contains(cache_key):
                        results[index] = process_cache_store.get(cache_key)
                    elif text not in missing_indices:
                        missing_indices[text] = [index]
                        missing_texts.append(text)
                    else:
                        missing_indices[text].append(index)

                if missing_texts:
                    task_results = client.predict_with_merge(missing_texts)
                    for text, result in zip(missing_texts, task_results):
                        # A reload may have advanced the generation while this
                        # lease was running. Such entries cannot be read by new
                        # requests, but can temporarily occupy cache capacity.
                        process_cache_store.add(_model_cache_key(generation, text), result)
                        for index in missing_indices[text]:
                            results[index] = result

                raw_matches = []
                for result in results:
                    raw_matches.extend(
                        pydantic_match_to_ml_match(match) for match in result.matches
                    )
            finally:
                _release_client(client)
            
            return ml_server_pb2.ProcessResponse(
                rawMatches=raw_matches,
                matches=raw_matches
            )
            
        except Exception as e:
            logger.error(f"Error in Process RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise


class MLServerServicer(ml_server_pb2_grpc.MLServerServicer):
    """gRPC servicer for LanguageTool MLServer."""

    def Match(self, request: ml_server_pb2.MatchRequest, context: grpc.ServicerContext) -> ml_server_pb2.MatchResponse:
        """
        Match grammar errors in sentences.
        
        Args:
            request: MatchRequest containing sentences to check
            context: gRPC context
            
        Returns:
            MatchResponse with matches for each sentence
        """
        try:
            logger.info(f"Match request: {len(request.sentences)} sentences")

            client, generation = _acquire_client()
            try:
                results = {i: None for i in range(len(request.sentences))}
                match_tasks = []  # (idx, text)
                for i, analyzed_sentence in enumerate(request.sentences):
                    cache_key = _model_cache_key(generation, analyzed_sentence)
                    if match_cache_store.contains(cache_key):
                        results[i] = match_cache_store.get(cache_key)
                    else:
                        match_tasks.append((i, analyzed_sentence))

                if match_tasks:
                    task_results = client.predict_with_merge([s for _, s in match_tasks])
                    for (idx, text), result in zip(match_tasks, task_results):
                        results[idx] = result
                        match_cache_store.add(_model_cache_key(generation, text), result)

                sentence_matches = []
                for i in range(len(request.sentences)):
                    enriched_result = results[i]
                    matches = ml_server_pb2.MatchList(
                        matches=[pydantic_match_to_ml_match(m) for m in enriched_result.matches]
                    )
                    sentence_matches.append(matches)
            finally:
                _release_client(client)
            return ml_server_pb2.MatchResponse(sentenceMatches=sentence_matches)
            
        except Exception as e:
            logger.error(f"Error in Match RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise

    def MatchAnalyzed(self, request: ml_server_pb2.AnalyzedMatchRequest, context: grpc.ServicerContext) -> ml_server_pb2.MatchResponse:
        """
        Match grammar errors in pre-analyzed sentences.
        
        Args:
            request: AnalyzedMatchRequest containing pre-analyzed sentences
            context: gRPC context
            
        Returns:
            MatchResponse with matches
        """
        try:
            logger.info(f"MatchAnalyzed request: {len(request.sentences)} analyzed sentences")

            client, generation = _acquire_client()
            try:
                results = {i: None for i in range(len(request.sentences))}
                match_tasks = []  # (idx, analyzed sentence)
                for i, analyzed_sentence in enumerate(request.sentences):
                    cache_key = _model_cache_key(generation, analyzed_sentence.text)
                    if match_anylized_cache_store.contains(cache_key):
                        results[i] = match_anylized_cache_store.get(cache_key)
                    else:
                        match_tasks.append((i, analyzed_sentence))

                if match_tasks:
                    task_results = client.predict_with_merge([s.text for _, s in match_tasks])
                    for (idx, analyzed_sentence), result in zip(match_tasks, task_results):
                        results[idx] = result
                        match_anylized_cache_store.add(
                            _model_cache_key(generation, analyzed_sentence.text), result
                        )

                sentence_matches = []
                for i in range(len(request.sentences)):
                    enriched_result = results[i]
                    matches = ml_server_pb2.MatchList(
                        matches=[pydantic_match_to_ml_match(m) for m in enriched_result.matches]
                    )
                    sentence_matches.append(matches)
            finally:
                _release_client(client)
            return ml_server_pb2.MatchResponse(sentenceMatches=sentence_matches)
            
        except Exception as e:
            logger.error(f"Error in MatchAnalyzed RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise


def serve(host: str = "0.0.0.0", port: int = 50051, health_port: int = 50052):
    """
    Start the gRPC server with ProcessingServer and MLServer services.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        health_port: Port for HTTP health check endpoint
    """
    global service_healthy

    initialize_runtime()
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=50))
    
    # Add servicers
    ml_server_pb2_grpc.add_ProcessingServerServicer_to_server(
        ProcessingServerServicer(), server
    )
    ml_server_pb2_grpc.add_MLServerServicer_to_server(
        MLServerServicer(), server
    )
    
    # Bind to port
    server.add_insecure_port(f"{host}:{port}")
    
    logger.info(f"Starting gRPC server on {host}:{port}")
    logger.info("Services:")
    logger.info("  - lt_ml_server.ProcessingServer (Analyze, Process)")
    logger.info("  - lt_ml_server.MLServer (Match, MatchAnalyzed)")
    server.start()
    
    # Start HTTP health check server in a background thread
    def run_http_health_server():
        health_server = HTTPServer((host, health_port), HealthCheckHTTPHandler)
        logger.info(f"Starting HTTP health check server on {host}:{health_port}")
        health_server.serve_forever()
    
    health_thread = Thread(target=run_http_health_server, daemon=True)
    health_thread.start()
    
    # Mark service as healthy after successful startup
    service_healthy = True
    logger.info("Service marked as healthy")
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        service_healthy = False
        server.stop(0)


if __name__ == "__main__":
    api_port = int(os.getenv('GRAMMARED_LANGUAGE__API_PORT', '50051'))
    api_host = os.getenv('GRAMMARED_LANGUAGE__API_HOST', '0.0.0.0')
    serve(host=api_host, port=api_port)
