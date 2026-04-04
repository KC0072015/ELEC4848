import { useCallback, useEffect, useRef, useState } from "react"
import { MapPin, Send, Trash2, Bot, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"
import { streamChat, clearSession, checkHealth, type ProximityNote } from "@/lib/api"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  proximity?: ProximityNote
  streaming?: boolean
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Health check on mount
  useEffect(() => {
    checkHealth().then(setApiOnline)
  }, [])

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const send = useCallback(() => {
    const text = input.trim()
    if (!text || isStreaming) return

    const userMsgId = crypto.randomUUID()
    const asstMsgId = crypto.randomUUID()

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: text },
      { id: asstMsgId, role: "assistant", content: "", streaming: true },
    ])
    setInput("")
    setIsStreaming(true)

    abortRef.current = streamChat(text, sessionId, {
      onSession: (id) => setSessionId(id),
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === asstMsgId ? { ...m, content: m.content + token } : m))
        )
      },
      onProximity: (note) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === asstMsgId ? { ...m, proximity: note } : m))
        )
      },
      onDone: () => {
        setMessages((prev) =>
          prev.map((m) => (m.id === asstMsgId ? { ...m, streaming: false } : m))
        )
        setIsStreaming(false)
      },
      onError: (detail) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId
              ? { ...m, content: `Error: ${detail}`, streaming: false }
              : m
          )
        )
        setIsStreaming(false)
      },
    })
  }, [input, isStreaming, sessionId])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const handleClear = async () => {
    if (isStreaming) {
      abortRef.current?.abort()
      setIsStreaming(false)
    }
    if (sessionId) await clearSession(sessionId)
    setSessionId(null)
    setMessages([])
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto w-full">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-[hsl(var(--border))]">
        <div className="flex items-center gap-2">
          <MapPin className="h-5 w-5 text-[hsl(var(--primary))]" />
          <span className="font-semibold text-[hsl(var(--foreground))]">HK Travel Guide</span>
          {apiOnline !== null && (
            <span
              className={cn(
                "ml-2 inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full",
                apiOnline
                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                  : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
              )}
            >
              <span className={cn("h-1.5 w-1.5 rounded-full", apiOnline ? "bg-green-500" : "bg-red-500")} />
              {apiOnline ? "online" : "offline"}
            </span>
          )}
        </div>
        <Button variant="ghost" size="icon" onClick={handleClear} title="New conversation">
          <Trash2 className="h-4 w-4" />
        </Button>
      </header>

      {/* Messages */}
      <ScrollArea className="flex-1 px-4">
        <div className="py-4 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-64 text-[hsl(var(--muted-foreground))] text-sm gap-2">
              <MapPin className="h-10 w-10 opacity-20" />
              <p>Ask me anything about Hong Kong!</p>
              <p className="opacity-60 text-xs">Try: "Best street food near Mong Kok"</p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={cn("flex gap-3 items-start", msg.role === "user" && "flex-row-reverse")}
            >
              <Avatar className="mt-0.5 shrink-0">
                <AvatarFallback className={cn(
                  msg.role === "assistant"
                    ? "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]"
                    : "bg-[hsl(var(--secondary))] text-[hsl(var(--secondary-foreground))]"
                )}>
                  {msg.role === "assistant" ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
                </AvatarFallback>
              </Avatar>

              <div className={cn("flex flex-col gap-1 max-w-[80%]", msg.role === "user" && "items-end")}>
                {/* Proximity badge */}
                {msg.proximity && (
                  <div className="flex items-center gap-1 text-xs text-[hsl(var(--muted-foreground))] bg-[hsl(var(--muted))] px-2 py-1 rounded-full self-start">
                    <MapPin className="h-3 w-3" />
                    {msg.proximity.count > 0
                      ? `${msg.proximity.count} attraction(s) within ${msg.proximity.threshold_km}km of ${msg.proximity.place}`
                      : `No attractions within ${msg.proximity.threshold_km}km of ${msg.proximity.place} — nearest: ${msg.proximity.closest} (${msg.proximity.closest_km}km)`}
                  </div>
                )}

                {/* Bubble */}
                <div
                  className={cn(
                    "rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap",
                    msg.role === "user"
                      ? "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] rounded-tr-sm"
                      : "bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] rounded-tl-sm"
                  )}
                >
                  {msg.content}
                  {msg.streaming && (
                    <span className="inline-block ml-0.5 animate-pulse">▌</span>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="px-4 py-3 border-t border-[hsl(var(--border))]">
        <div className="flex gap-2 items-end">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about Hong Kong attractions, food, itineraries…"
            className="resize-none min-h-[44px] max-h-36"
            rows={1}
            disabled={!apiOnline}
          />
          <Button
            size="icon"
            onClick={send}
            disabled={!input.trim() || isStreaming || !apiOnline}
            className="shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-xs text-[hsl(var(--muted-foreground))] mt-1.5 text-center">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
