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
    {"name": "Alibaba Cloud Blog", "url": "https://www.alibabacloud.com/blog/feed"},
    {"name": "Alibaba Developer Blog", "url": "https://developer.aliyun.com/rss.xml"},
    {"name": "Tencent Cloud Developer", "url": "https://cloud.tencent.com/developer/rss"},
    {"name": "Tencent Open Source", "url": "https://opensource.tencent.com/feed"},
    {"name": "Huawei Developer Blog", "url": "https://developer.huawei.com/ict/en/blog/rss"},
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
    forbidden = False
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
            return resp.text, False
        if resp.status_code == 403:
            forbidden = True
            print("Jina Reader 403，尝试直连原站...")
    except Exception:
        pass
    if forbidden:
        return None, True
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
            return direct.text, False
        if direct.status_code == 403:
            return None, True
    except Exception:
        pass
    if fallback_summary:
        return fallback_summary, False
    return "（无法获取正文，请基于标题总结）", False


def summarize_daily_brief(client, articles):
    """整合当天文章，一次性生成简报与趋势洞察"""
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

    items = []
    for article in articles:
        print(f"正在整理: {article['title']}...")
        content, forbidden = fetch_content_with_jina(
            article["url"], article.get("summary")
        )
        if forbidden:
            print(f"403 跳过文章: {article['title']}")
            continue

        text = content or article.get("summary") or ""
        if not text:
            continue
        items.append(
            {
                "title": article["title"],
                "source": article["source"],
                "url": article["url"],
                "text": text[:6000],
            }
        )

    if not items:
        return None

    merged = "\n\n".join(
        [
            f"【{item['source']}】{item['title']}\n"
            f"链接: {item['url']}\n"
            f"内容:\n{item['text']}"
            for item in items
        ]
    )

    prompt = f"""
    你是一个 AI 技术情报专家。请基于以下多篇文章，整合生成当天简报。
    重点：不要逐篇复述，务必提炼跨来源的趋势性洞察与共性信号。

    ---
    资料（共 {len(items)} 篇）：
    {merged}

    ---
    请输出严格的 Markdown（不要使用代码块），结构如下：

    # 今日 AI 简报
    ## 今日要点
    - (3-6 条，跨来源汇总)

    ## 大厂动态
    - **公司**：1-2 句概括关键更新

    ## 技术趋势
    - (3-5 条趋势洞察，强调变化与影响)

    ## 文章索引
    - [标题](链接) — 来源
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings,
            ),
        )
        return response.text if response else None
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None


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

    report = summarize_daily_brief(client, articles)
    if report:
        send_to_feishu(report)


if __name__ == "__main__":
    main()
