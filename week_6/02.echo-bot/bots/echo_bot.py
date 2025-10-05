# bots/echo_bot.py
import os, logging
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ClientAuthenticationError
from azure.ai.textanalytics.aio import TextAnalyticsClient

logging.basicConfig(level=logging.DEBUG)                 # DEBUG while diagnosing
logging.getLogger("azure").setLevel(logging.DEBUG)       # see Azure SDK logs

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip()
KEY = os.getenv("AZURE_LANGUAGE_KEY", "").strip()

# Validate config early (clear error instead of a mystery crash later)
if not ENDPOINT or not KEY:
    raise RuntimeError("Missing AZURE_LANGUAGE_ENDPOINT or AZURE_LANGUAGE_KEY")

# Enable HTTP logging on the client (redacts secrets, but still be careful with logs)
client = TextAnalyticsClient(ENDPOINT, AzureKeyCredential(KEY), logging_enable=True)

def bin_label(pos, neg, neu):
    if pos >= 0.60 and pos > neg: return "positive"
    if neg >= 0.60 and neg > pos: return "negative"
    return "neutral"

class EchoBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        text = (turn_context.activity.text or "").strip()
        if not text:
            await turn_context.send_activity("Please send some text.")
            return
        try:
            # per-call HTTP logging (optional, extra verbose)
            result = await client.analyze_sentiment([text], logging_enable=True)
            doc = result[0]
            if getattr(doc, "is_error", False):
                # Show service error code/message from the result item
                err = doc.error
                await turn_context.send_activity(f"Azure error: {err.code}: {err.message}")
                return

            s = doc.confidence_scores
            label = bin_label(s.positive, s.negative, s.neutral)
            await turn_context.send_activity(
                MessageFactory.text(f"Sentiment: {label} (pos={s.positive:.2f}, neu={s.neutral:.2f}, neg={s.negative:.2f})")
            )

        except ClientAuthenticationError as e:
            # 401/403 – bad key or wrong resource
            await turn_context.send_activity("Auth error calling Azure Language (check key/endpoint).")
            raise
        except HttpResponseError as e:
            # Service returned an HTTP error (400/404/415/429/5xx)
            # Log details; many responses include helpful JSON in e.response
            logging.exception("HttpResponseError status=%s", getattr(e, "status_code", "?"))
            await turn_context.send_activity(f"Azure Language HTTP error {getattr(e, 'status_code', '?')}.")
            raise
        except Exception:
            logging.exception("Unexpected failure")
            await turn_context.send_activity("Unexpected error while analyzing sentiment.")
            raise
