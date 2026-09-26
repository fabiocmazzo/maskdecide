
import base64
import gzip
import json
import logging
import time
import traceback

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger("http_logger")


def format_body(
    body: bytes,
    content_type: str = "",
    content_encoding: str = "",
):
    """Formata JSON, texto ou conteúdo binário."""

    if not body:
        return None

    # Descompacta respostas comprimidas, quando possível.
    if "gzip" in content_encoding.lower():
        try:
            body = gzip.decompress(body)
        except (OSError, EOFError):
            pass

    # Tenta interpretar como texto UTF-8.
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "type": "binary",
            "size_bytes": len(body),
            "base64": base64.b64encode(body).decode("ascii"),
        }

    # Formata JSON como objeto, permitindo indentação.
    if (
        "application/json" in content_type.lower()
        or "+json" in content_type.lower()
    ):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Também identifica JSON quando o Content-Type está incorreto.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class HTTPLoggerMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):


        # Ignora WebSocket e eventos de inicialização.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()

        request = Request(scope, receive=receive)

        request_body = b""
        response_body = bytearray()

        response_status = None
        response_headers = {}

        error = None

        try:
            # Lê o body completo antes de executar a aplicação.
            request_body = await request.body()

            # Reconstrói o canal de leitura para que a aplicação
            # receba exatamente os mesmos bytes.
            body_replayed = False

            async def receive_wrapper() -> Message:
                nonlocal body_replayed

                if not body_replayed:
                    body_replayed = True

                    return {
                        "type": "http.request",
                        "body": request_body,
                        "more_body": False,
                    }

                # Mantém o comportamento normal do canal ASGI,
                # incluindo eventos de desconexão.
                return await receive()

            # Intercepta mensagens enviadas pela aplicação.
            async def send_wrapper(message: Message):
                nonlocal response_status
                nonlocal response_headers

                if message["type"] == "http.response.start":
                    response_status = message["status"]

                    response_headers = {
                        key.decode("latin-1"): value.decode("latin-1")
                        for key, value in message.get("headers", [])
                    }

                elif message["type"] == "http.response.body":
                    response_body.extend(message.get("body", b""))

                # Encaminha a mensagem original sem alterações.
                await send(message)

            await self.app(
                scope,
                receive_wrapper,
                send_wrapper,
            )

        except Exception:
            error = traceback.format_exc()

            if response_status is None:
                response_status = 500

            raise

        finally:
            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            request_headers = dict(request.headers)

            log_data = {
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "client": (
                        request.client.host
                        if request.client
                        else None
                    ),
                    "headers": request_headers,
                    "query_params": dict(
                        request.query_params.multi_items()
                    ),
                    "body": format_body(
                        request_body,
                        request_headers.get(
                            "content-type", ""
                        ),
                        request_headers.get(
                            "content-encoding", ""
                        ),
                    ),
                },
                "response": {
                    "status": response_status,
                    "headers": response_headers,
                    "body": format_body(
                        bytes(response_body),
                        response_headers.get(
                            "content-type", ""
                        ),
                        response_headers.get(
                            "content-encoding", ""
                        ),
                    ),
                },
                "duration_ms": round(duration_ms, 2),
            }

            if error:
                log_data["exception"] = error

            #formatted_log = json.dumps(
             #   log_data,
             #   indent=2,
             #   ensure_ascii=False,
             #   default=str,
            #)

            #logger.info(
            #    "\n%s\n",
            #    formatted_log,
            #)