import os
import datetime
import time
import requests
import feedparser
from google import genai
from google.genai import types

# 配置
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 默认使用 gemini-2.5-flash，速度快且免费额度足够
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.5-flash"

# RSS 源列表
RSS_SOURCES = [
    {"name": "OpenAI", "url": "https://openai.com/news/rss.xml"},
    {"name": "Anthropic", "url": "https://www.anthropic.com/feed"},
    {"name": "Google DeepMind", "url": "https://deepmind.google/blog/rss.xml"},
    {"name": "Hugging Face", "url": "https://huggingface.co/blog/feed.xml"},
    {"name": "Meta AI", "url": "https://ai.meta.com/blog/rss.xml"},
    {"name": "LangChain", "url": "https://blog.langchain.dev/rss/"},
    {"name": "Microsoft Research", "url": "https://www.microsoft.com/en-us/research/feed/"},
    {"name": "Apple Machine Learning", "url": "https://machinelearning.apple.com/rss.xml"},
    {"name": "AWS Machine Learning", "url": "https://aws.amazon.com/blogs/machine-learning/feed/"},
    {"name": "Google AI Blog", "url": "https://ai.googleblog.com/feeds/posts/default"},
    {"name": "OpenAI Research (arXiv)", "url": "https://export.arxiv.org/rss/cs.AI"},
    {"name": "Machine Learning (arXiv)", "url": "https://export.arxiv.org/rss/cs.LG"},
    {"name": "Papers With Code", "url": "https://paperswithcode.com/rss"},
]


def get_recent_articles():
    """抓取过去 24 小时的文章"""
    print("正在抓取 RSS 源...")
    recent_articles = []
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=24)

    for source in RSS_SOURCES:
        try:
            print(f"正在检查: {source['name']}...")
            feed = feedparser.parse(source["url"])
            if not feed.entries:
                continue

            for entry in feed.entries:
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime.datetime(
                        *entry.published_parsed[:6],
                        tzinfo=datetime.timezone.utc,
                    )
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime.datetime(
                        *entry.updated_parsed[:6],
                        tzinfo=datetime.timezone.utc,
                    )

                if pub_date and pub_date > cutoff:
                    print(f"  [发现新文章] {entry.title}")
                    recent_articles.append(
                        {
                            "title": entry.title,
                            "url": entry.link,
                            "source": source["name"],
                            "date": pub_date.strftime("%Y-%m-%d"),
                            "summary": getattr(entry, "summary", None)
                            or getattr(entry, "description", None),
                        }
                    )
        except Exception as e:
            print(f"Error fetching {source['name']}: {e}")
            continue

    return recent_articles


def fetch_content_with_jina(url, fallback_summary=None):
    """使用 Jina Reader 获取正文，必要时回退到原站或摘要"""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = requests.get(
            jina_url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AINewsBot/1.0)",
                "Referer": url,
            },
        )
        if resp.status_code == 200 and resp.text:
            return resp.text
        if resp.status_code == 403:
            print("Jina Reader 403，尝试直连原站...")
    except Exception:
        pass
    try:
        direct = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AINewsBot/1.0)",
                "Referer": url,
            },
        )
        if direct.status_code == 200 and direct.text:
            return direct.text
    except Exception:
        pass
    if fallback_summary:
        return fallback_summary
    return "（无法获取正文，请基于标题总结）"


def summarize_article(client, article):
    """调用 Gemini 总结单篇文章"""
    print(f"正在总结: {article['title']}...")
    content = fetch_content_with_jina(article["url"], article.get("summary"))

    # Gemini 1.5 窗口很大，我们可以保留更多内容 (30k chars 约 10k tokens，安全)
    content_snippet = content[:30000]

    prompt = f"""
    你是一个 AI 技术情报专家。请阅读以下技术博客内容，为中文读者生成这篇简报。

    文章标题: {article['title']}
    来源: {article['source']}
    内容:
    {content_snippet}

    ---
    请输出严格的 Markdown 格式总结（不要使用代码块包裹）：

    **{article['source']}** · [{article['title']}]({article['url']})
    > 💡 **核心观点**: (一句话概括核心发布或研究成果)
    > 🎯 **关键技术**: (列出 2-3 个关键技术点/参数/性能提升)
    > 🔮 **影响**: (一句话点评对开发者或行业的影响)
    """

    def _generate(contents):
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings,
            ),
        )

    try:
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        response = _generate(prompt)
        text = response.text if response else None
        if text and "无法为您生成这篇简报" not in text:
            return text

        # 回退：仅使用标题/摘要，避免正文抓取失败或触发拒答
        summary_fallback = article.get("summary") or "（无摘要）"
        short_prompt = f"""
        你是一个 AI 技术情报专家。请仅基于标题与摘要，为中文读者生成简报。

        文章标题: {article['title']}
        来源: {article['source']}
        摘要:
        {summary_fallback}

        ---
        请输出严格的 Markdown 格式总结（不要使用代码块包裹）：

        **{article['source']}** · [{article['title']}]({article['url']})
        > 💡 **核心观点**: (一句话概括核心发布或研究成果)
        > 🎯 **关键技术**: (列出 2-3 个关键技术点/参数/性能提升)
        > 🔮 **影响**: (一句话点评对开发者或行业的影响)
        """
        response = _generate(short_prompt)
        return response.text if response else "（AI 总结失败）"
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"**{article['title']}**\n> (AI 总结失败: {str(e)})"


def send_to_feishu(content):
    """发送汇总到飞书"""
    if not FEISHU_WEBHOOK:
        print("未配置 FEISHU_WEBHOOK，只打印内容：")
        print(content)
        return

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "template": "blue",
                "title": {
                    "content": f"🚀 AI 每日速递 ({today})",
                    "tag": "plain_text",
                },
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": content if content else "今日前沿平静，暂无重大发布。",
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": "Powered by Gemini 1.5 & GitHub Actions",
                        }
                    ],
                },
            ],
        },
    }

    requests.post(FEISHU_WEBHOOK, json=card)
    print("已推送到飞书")


def main():
    if not GEMINI_API_KEY:
        print("Error: 请设置 GEMINI_API_KEY")
        return

    # 初始化 Gemini (google-genai 新 SDK)
    client = genai.Client(api_key=GEMINI_API_KEY)

    articles = get_recent_articles()

    if not articles:
        print("今日无新文章")
        return

    summaries = []
    for art in articles:
        summary = summarize_article(client, art)
        if summary:
            summaries.append(summary)
        # Gemini 速率限制宽松 (Flash 版 15 RPM)，基本不需要 sleep，但安全起见休眠 2s
        time.sleep(2)

    if summaries:
        final_report = "\n\n---\n\n".join(summaries)
        send_to_feishu(final_report)


if __name__ == "__main__":
    main()
