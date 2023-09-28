from slack_sdk.webhook import WebhookClient
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def send_slack_message(blocks, token):
    """Sends a notification to slack. Message are send by blocks of 20 to limit oversized message."""
    url = f"https://hooks.slack.com/services/{token}"
    webhook = WebhookClient(url)
    chunked_blocks = [blocks[i:i + 20] for i in range(0, len(blocks), 20)]
    for chunk_block in chunked_blocks:
        response = webhook.send(
            text="fallback",
            blocks=chunk_block
        )
        if response.status_code != 200:
            raise ValueError(f'Request to slack returned an error {response.status_code}, '
                             f'the response is:\n{response.body}')


def send_msg_to_slack_channel(msg: str, channel_id: str, token: str) -> None:
    client = WebClient(token=token)
    try:
        client.chat_postMessage(
            channel=channel_id,
            text=msg
        )
    except SlackApiError as e:
        raise ValueError(str(e))
    return


def send_file_to_slack_channel(file_path: str, title: str, channel_id: str, token: str):
    client = WebClient(token=token)
    try:
        client.files_upload(
            channels=channel_id,
            file=file_path,
            title=title
        )
    except SlackApiError as e:
        ValueError(str(e))
    return
