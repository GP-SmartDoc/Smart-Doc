import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { Send, Upload, ChevronLeft, ChevronDown, MessageSquare, FileText, Layout, Share2, Plus, Zap, Cpu, Settings, LogOut, Menu, X, Copy, Check, Search, MoreVertical, Edit2, Trash2 } from 'lucide-react';
import { useAuth } from '../lib/AuthContext';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: true,
  theme: "dark",
  themeVariables: {
    background: "#1a1d25",
    primaryColor: "#1f2937",
    primaryTextColor: "#ffffff",
    primaryBorderColor: "#ffffff",
    lineColor: "#ffffff",
    textColor: "#ffffff"
  },
  flowchart: {
    htmlLabels: true,
    useMaxWidth: false,
  }
});

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const MermaidChart = ({ chart }: { chart: string }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartId = useRef(`mermaid-${Date.now()}-${Math.floor(Math.random() * 10000)}`).current;
  
  useEffect(() => {
    let isMounted = true;
    if (chartRef.current) {
      const cleanChart = chart.replace(/%%\{[\s\S]*?\}%%/g, '').trim();
      mermaid.render(chartId, cleanChart)
        .then(({ svg }) => {
          if (isMounted && chartRef.current) {
            chartRef.current.innerHTML = svg;
          }
        })
        .catch(e => {
          console.error("Mermaid error", e);
          if (isMounted && chartRef.current) {
             // To prevent showing massive stack trace text from mermaid in the UI
             chartRef.current.innerHTML = `<div class="text-red-400 p-4 border border-red-500/30 rounded-lg bg-red-500/10">Failed to render diagram (Syntax Error).</div>`;
          }
        });
    }
    return () => { isMounted = false; };
  }, [chart, chartId]);

  return <div ref={chartRef} className="w-full overflow-x-auto rounded-xl bg-[#1a1d25] p-4 text-white min-h-[200px] [&_svg]:mx-auto" />;
};

const AIMessageRender = ({ text, isStreaming, onComplete }: { text: string, isStreaming?: boolean, onComplete?: () => void }) => {
  // Strip any mermaid init blocks the LLM might hallucinate outside code blocks
  const sanitizedText = text.replace(/%%\{init:[\s\S]*?\}%%/g, '');

  if (sanitizedText.includes('```mermaid')) {
    const parts = sanitizedText.split('```mermaid');
    const preText = parts[0];
    const rest = parts[1] || '';
    const codeParts = rest.split('```');
    const chartCode = codeParts[0].trim();
    const postText = codeParts.slice(1).join('```');

    if (isStreaming && onComplete) onComplete();

    return (
      <div className="space-y-4 w-full">
        {preText.trim() && <div dangerouslySetInnerHTML={{ __html: preText.trim().replace(/\n/g, '<br/>') }} />}
        {chartCode && <MermaidChart chart={chartCode} />}
        {postText.trim() && <div dangerouslySetInnerHTML={{ __html: postText.trim().replace(/\n/g, '<br/>') }} />}
      </div>
    );
  }
  
  if (isStreaming) {
    return <Typewriter text={sanitizedText} onComplete={onComplete} />;
  }
  
  return <div dangerouslySetInnerHTML={{ __html: sanitizedText.replace(/\n/g, '<br/>') }} className="break-words w-full" />;
};

const Typewriter = ({ text, onComplete }: { text: string; onComplete?: () => void }) => {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    let i = 0;
    const t = setInterval(() => {
      setDisplayed(prev => prev + text.charAt(i));
      i++;
      if (i >= text.length) {
        clearInterval(t);
        onComplete?.();
      }
    }, 15);
    return () => clearInterval(t);
  }, [text, onComplete]);
  return <div className="inline" dangerouslySetInnerHTML={{ __html: displayed.replace(/\n/g, '<br/>') + '<span class="inline-block w-1.5 h-3 ml-1 align-middle bg-[#2563eb] animate-pulse"></span>' }} />;
};

type Message = {
  id: string;
  type: 'user' | 'ai';
  text: string;
  timestamp: Date;
  documentName?: string;
  isStreaming?: boolean;
};

type ChatSession = {
  id: string;
  title: string;
  updatedAt: Date;
  messages: Message[];
  activeFileIndex: number | null;
  files: { name: string; size: string }[];
};

type Mode = 'QA' | 'Summary' | 'Slides' | 'Viz';
type SummarySubMode = 'Snapshot' | 'Overview' | 'Deep Dive';

export default function ChatApp() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', type: 'ai', text: `Hello ${user?.displayName || 'there'}! I'm SmartDoc. Upload a PDF and I'll help you analyze it. What would you like to do?`, timestamp: new Date() }
  ]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<Mode>('QA');
  const [summarySubMode, setSummarySubMode] = useState<SummarySubMode>('Overview');
  const [files, setFiles] = useState<{ name: string; size: string }[]>([]);
  const [activeFileIndex, setActiveFileIndex] = useState<number | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth >= 768);
  const [isDragging, setIsDragging] = useState(false);
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>(Date.now().toString());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = () => setOpenMenuId(null);
    if (openMenuId) {
      document.addEventListener('click', handleClickOutside);
    }
    return () => document.removeEventListener('click', handleClickOutside);
  }, [openMenuId]);

  const deleteChat = (chatId: string) => {
    setChats(prev => {
      const next = prev.filter(c => c.id !== chatId);
      if (user) {
        localStorage.setItem(`smartdoc_chats_${user.uid}`, JSON.stringify(next));
      }
      return next;
    });
    if (currentChatId === chatId) {
      createNewChat();
    }
  };

  const renameChat = (chatId: string, newTitle: string) => {
    setChats(prev => {
      const next = prev.map(c => c.id === chatId ? { ...c, title: newTitle || c.title } : c);
      if (user) {
        localStorage.setItem(`smartdoc_chats_${user.uid}`, JSON.stringify(next));
      }
      return next;
    });
    setEditingChatId(null);
  };

  useEffect(() => {
    if (!user) {
      setChats([]);
      return;
    }
    const key = `smartdoc_chats_${user.uid}`;
    const saved = localStorage.getItem(key);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        parsed.forEach((c: any) => {
          c.updatedAt = new Date(c.updatedAt);
          c.messages.forEach((m: any) => {
             m.timestamp = new Date(m.timestamp);
             m.isStreaming = false;
          });
        });
        setChats(parsed);
        if (parsed.length > 0) {
          setCurrentChatId(parsed[0].id);
          setMessages(parsed[0].messages);
          setFiles(parsed[0].files || []);
          setActiveFileIndex(parsed[0].activeFileIndex);
        }
      } catch (e) {
        console.error("Failed parsing history", e);
      }
    }
  }, [user?.uid]);

  useEffect(() => {
    setChats(prev => {
      const exists = prev.findIndex(c => c.id === currentChatId);
      const existingTitle = exists >= 0 ? prev[exists].title : null;
      
      let defaultTitle = 'New Chat';
      if (files.length > 0) defaultTitle = files[0].name;
      else if (messages.length > 1) {
        const firstUserMsg = messages.find(m => m.type === 'user');
        if (firstUserMsg) defaultTitle = firstUserMsg.text.slice(0, 25);
      }

      const newChat: ChatSession = {
        id: currentChatId,
        title: existingTitle && existingTitle !== 'New Chat' ? existingTitle : defaultTitle,
        updatedAt: new Date(),
        messages,
        files,
        activeFileIndex
      };

      const next = [...prev];
      if (exists >= 0) {
         next[exists] = newChat;
      } else {
         next.unshift(newChat);
      }
      if (user) {
        localStorage.setItem(`smartdoc_chats_${user.uid}`, JSON.stringify(next));
      }
      return next;
    });
  }, [messages, files, activeFileIndex, currentChatId, user?.uid]);

  const loadChat = (chat: ChatSession) => {
    setCurrentChatId(chat.id);
    setMessages(chat.messages);
    setFiles(chat.files);
    setActiveFileIndex(chat.activeFileIndex);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const createNewChat = () => {
    if (messages.length <= 1 && files.length === 0) {
      if (window.innerWidth < 768) setIsSidebarOpen(false);
      return;
    }
    const newId = Date.now().toString();
    setCurrentChatId(newId);
    setMessages([{ id: newId, type: 'ai', text: `Hello ${user?.displayName || 'there'}! I'm SmartDoc. Upload a PDF and I'll help you analyze it. What would you like to do?`, timestamp: new Date() }]);
    setFiles([]);
    setActiveFileIndex(null);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/');
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const handleCopy = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text', err);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const documentName = activeFileIndex !== null ? files[activeFileIndex].name : "all";
    const userMsg: Message = {
      id: Date.now().toString(),
      type: 'user',
      text: input,
      timestamp: new Date(),
      documentName
    };

    setMessages(prev => [...prev, userMsg]);
    const currentInput = input;
    setInput('');
    setIsTyping(true);

    try {
      let pyMode = 'qa';
      if (mode === 'QA') pyMode = 'qa';
      if (mode === 'Summary') pyMode = 'summary';
      if (mode === 'Slides') pyMode = 'slide_generation';
      if (mode === 'Viz') pyMode = 'visualization';

      let pySummaryMode = null;
      if (mode === 'Summary') {
        if (summarySubMode === 'Snapshot') pySummaryMode = 'snapshot';
        if (summarySubMode === 'Overview') pySummaryMode = 'overview';
        if (summarySubMode === 'Deep Dive') pySummaryMode = 'deepdive';
      }

      const payload: any = {
        message: currentInput,
        document: documentName,
        mode: pyMode,
      };
      if (pySummaryMode) payload.summary_mode = pySummaryMode;

      const headers: any = { "Content-Type": "application/json" };
      if (user?.uid) headers['X-User-ID'] = user.uid;
      const res = await fetch(`${API_BASE_URL}/send`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      const safeDecodeUnicode = (str: string) => {
        if (/\\u[0-9a-fA-F]{4}/.test(str)) {
          return str.replace(/\\u[0-9a-fA-F]{4}/g, function(match) {
            return String.fromCharCode(parseInt(match.replace("\\u", ""), 16));
          });
        }
        return str;
      };

      const formattedReply = safeDecodeUnicode(data.reply);
      
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        text: formattedReply,
        timestamp: new Date(),
        documentName,
        isStreaming: true
      };
      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        type: 'ai',
        text: `Error communicating with server. Ensure backend is running at ${API_BASE_URL}.`,
        timestamp: new Date(),
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const getMockResponse = (query: string, currentMode: Mode, subMode: SummarySubMode) => {
    if (currentMode === 'Summary') {
      if (subMode === 'Snapshot') return "Snapshot Summary: This document explores the intersection of AI and document intelligence, focusing on 3 key pillars: RAG, Semantic Search, and LLM orchestration.";
      if (subMode === 'Deep Dive') return "Detailed Analysis (Deep Dive):\n\nChapter 1 highlights the historical context of PDF processing...\n\nChapter 2 introduces the 'SmartDoc' architecture which solves the CTRL+F limitation by using vector embeddings...\n\nConclusion: The future of document intelligence lies in conversational agents that reason over grounded data.";
      return "Overview: The text focuses on the strategic implementation of AI in enterprise workflows, highlighting efficiency gains of 30% while noting initial integration challenges.";
    }
    if (currentMode === 'Slides') return "I've generated a slide outline for you: \n1. Executive Summary\n2. The Core Problem\n3. Our AI Solution\n4. Implementation Roadmap\n5. Q&A Session";
    if (currentMode === 'Viz') return "Generating Diagram...\n\n[Mermaid Chart Created: Process flow from Data Input -> Cleaning -> Vector Storage -> Querying]";
    return `I found information related to "${query}" in Section 3.2. It states that the primary objectives are aligned with the 2025 sustainability goals.`;
  };

  const processFiles = async (fileList: FileList | File[]) => {
    const filesArray = Array.from(fileList);
    const validFiles = filesArray;
    if (validFiles.length === 0) return;

    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      const sizeStr = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
      
      setFiles(prev => {
        const exists = prev.find(f => f.name === file.name);
        if (exists) return prev;
        const next = [...prev, { name: file.name, size: sizeStr }];
        setTimeout(() => setActiveFileIndex(next.length - 1), 0);
        return next;
      });

      // Show user message for file submission like the old html does
      const fileMsg: Message = {
        id: 'file-' + Date.now() + '-' + i,
        type: 'user',
        text: `📎 ${file.name}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, fileMsg]);
      setIsTyping(true);

      const formData = new FormData();
      formData.append("files", file);

      try {
        const headers: any = {};
        if (user?.uid) headers['X-User-ID'] = user.uid;
        const res = await fetch(`${API_BASE_URL}/upload`, {
          method: "POST",
          headers,
          body: formData
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        let replyText = `❌ Upload failed for ${file.name}. Debug: ${JSON.stringify(data)}`;
        const isUploaded = data.uploaded && data.uploaded.some((item: any) => 
          typeof item === 'string' ? item === file.name : item.filename === file.name
        );
        const isSkipped = data.skipped && data.skipped.some((item: any) => 
          typeof item === 'string' ? item === file.name : item.filename === file.name
        );

        if (isUploaded) {
          replyText = `✅ Uploaded ${file.name} successfully`;
        } else if (isSkipped) {
          replyText = `Already indexed ${file.name}`;
        }

        setMessages(prev => [...prev, {
          id: 'upload-' + Date.now() + '-' + i,
          type: 'ai',
          text: replyText,
          timestamp: new Date()
        }]);

        // Refresh documents list
        loadPDFs();
      } catch (error: any) {
        console.error(error);
        setMessages(prev => [...prev, {
          id: 'upload-' + Date.now() + '-' + i,
          type: 'ai',
          text: `❌ Upload failed for ${file.name}. Error: ${error.message || error}`,
          timestamp: new Date()
        }]);
      } finally {
        setIsTyping(false);
      }
    }
  };

  const loadPDFs = async () => {
    try {
      const headers: any = {};
      if (user?.uid) headers['X-User-ID'] = user.uid;
      const res = await fetch(`${API_BASE_URL}/documents`, { headers });
      const data = await res.json();
      if (data.documents) {
        setFiles(data.documents.map((doc: string) => ({ name: doc, size: 'Unknown' })));
      }
    } catch (error) {
      console.error("Failed to load PDFs:", error);
    }
  };

  useEffect(() => {
    loadPDFs();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="flex h-screen bg-[#0f1117] text-[#e5e7eb] font-sans overflow-hidden"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 z-[9999] bg-[#2563eb]/20 backdrop-blur-sm border-4 border-dashed border-[#2563eb] rounded-2xl flex flex-col items-center justify-center pointer-events-none transition-all m-4">
          <div className="w-24 h-24 bg-[#2563eb] rounded-full flex items-center justify-center mb-6 shadow-[0_0_50px_rgba(37,99,235,0.5)] animate-bounce">
            <Upload size={40} className="text-white" />
          </div>
          <h2 className="text-4xl font-extrabold text-white tracking-tighter mb-2">Drop your PDF</h2>
          <p className="text-[#e5e7eb] text-xl">We'll instantly analyze it.</p>
        </div>
      )}
      {/* Hidden File Input */}
      <input 
        type="file" 
        multiple
        ref={fileInputRef} 
        onChange={handleFileChange} 
        className="hidden" 
      />
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed md:relative inset-y-0 left-0 z-50 bg-[#111318] border-r border-white/5 flex flex-col overflow-hidden transition-all duration-300 ease-in-out flex-shrink-0 ${isSidebarOpen ? 'w-64 translate-x-0' : 'w-0 -translate-x-full md:translate-x-0'}`}>
        <div className="p-6 border-b border-white/5 flex items-center justify-between shadow-sm flex-shrink-0 w-64">
          <Link to="/" className="text-xl font-black text-white tracking-tighter flex items-center">
            S<span className="tracking-tight mr-[1px]">mart</span><span className="text-[#2563eb]">D</span><span className="text-[#2563eb] tracking-tight">oc</span>
          </Link>
          <button 
            type="button"
            onClick={() => setIsSidebarOpen(false)}
            className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors md:hidden"
          >
            <X size={16} className="text-[#6b7280]" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 w-64 min-h-0 flex flex-col gap-6">
          <div>
             <button 
               type="button"
               onClick={createNewChat}
               className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#2563eb] text-white rounded-xl font-bold text-sm hover:scale-[1.02] transition-transform shadow-lg shadow-blue-900/10"
              >
              <Plus size={18} /> New Chat
             </button>
          </div>

          {chats.length > 0 && (
          <div>
             <p className="text-[10px] text-[#6b7280] font-bold uppercase tracking-widest px-4 mb-4">Recent Chats</p>
             <div className="space-y-1">
               {chats.slice(0, 15).map(c => (
                 <div key={c.id} className={`group relative w-full flex items-center justify-between px-2 py-1.5 rounded-xl text-xs transition-colors ${currentChatId === c.id ? 'bg-[#2563eb]/10 text-[#2563eb] font-bold' : 'text-[#6b7280] hover:text-white hover:bg-white/5'}`}>
                   {editingChatId === c.id ? (
                     <input
                       type="text"
                       value={editingTitle}
                       onChange={(e) => setEditingTitle(e.target.value)}
                       onKeyDown={(e) => {
                         if (e.key === 'Enter') renameChat(c.id, editingTitle);
                         if (e.key === 'Escape') setEditingChatId(null);
                       }}
                       onBlur={() => renameChat(c.id, editingTitle)}
                       autoFocus
                       className="w-full bg-transparent border border-[#2563eb] rounded px-2 py-1 outline-none text-white text-xs"
                     />
                   ) : (
                     <>
                       <button
                         onClick={() => loadChat(c)}
                         className="flex-1 text-left truncate px-2 py-1"
                       >
                         {c.title}
                       </button>
                       <div className="relative">
                         <button 
                           onClick={(e) => {
                             e.stopPropagation();
                             setOpenMenuId(openMenuId === c.id ? null : c.id);
                           }}
                           className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded-md transition-opacity"
                         >
                           <MoreVertical size={14} />
                         </button>
                         {openMenuId === c.id && (
                           <div className="absolute right-0 top-full mt-1 w-32 bg-[#1a1d25] border border-white/10 rounded-lg shadow-xl z-50 overflow-hidden"
                                onClick={(e) => e.stopPropagation()}
                           >
                             <button
                               onClick={() => {
                                 setEditingChatId(c.id);
                                 setEditingTitle(c.title);
                                 setOpenMenuId(null);
                               }}
                               className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-white hover:bg-white/5"
                             >
                               <Edit2 size={12} /> Rename
                             </button>
                             <button
                               onClick={() => {
                                 deleteChat(c.id);
                                 setOpenMenuId(null);
                               }}
                               className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-red-500 hover:bg-white/5"
                             >
                               <Trash2 size={12} /> Delete
                             </button>
                           </div>
                         )}
                       </div>
                     </>
                   )}
                 </div>
               ))}
             </div>
          </div>
          )}
        </div>

        <div className="p-4 border-t border-white/5 w-64 mt-auto flex-shrink-0 bg-[#0a0c10]/50">
           <div className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 transition-colors rounded-xl border border-white/5">
              {user?.photoURL ? (
                <img src={user.photoURL} alt={user.displayName || ''} className="w-8 h-8 rounded-full" referrerPolicy="no-referrer" />
              ) : (
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#2563eb] to-[#7c3aed] flex items-center justify-center text-white text-[10px] font-bold">
                  {user?.displayName?.split(' ').map(n => n[0]).join('') || 'U'}
                </div>
              )}
              <div className="flex-1 overflow-hidden">
                <p className="text-xs font-bold text-white truncate">{user?.displayName || 'User'}</p>
                <p className="text-[10px] text-[#6b7280] truncate">{user?.email}</p>
              </div>
              <button 
                onClick={handleLogout}
                className="p-1.5 hover:bg-white/5 rounded-lg text-[#6b7280] hover:text-red-500 transition-colors"
                title="Logout"
              >
                <LogOut size={14} />
              </button>
           </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative">
        {/* Header */}
        <header className="h-[70px] flex-shrink-0 border-b border-white/5 flex items-center justify-between px-4 md:px-8 bg-[#0f1117] z-30">
           <div className="flex items-center gap-3 md:gap-4">
              <button 
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="p-2 -ml-2 text-[#6b7280] hover:text-white transition-colors"
              >
                <Menu size={24} />
              </button>
              <Link to="/" className="hidden md:flex items-center text-lg font-black text-white pr-4 border-r border-white/5 tracking-tighter">
                S<span className="text-[#2563eb]">D</span>
              </Link>
              <div className="flex flex-col">
                {files.length > 0 ? (
                  <div className="relative flex items-center group cursor-pointer -mx-1 px-1 rounded hover:bg-white/5">
                    <span className="text-sm font-bold text-white pr-5 truncate max-w-[150px] md:max-w-[250px]">
                      {activeFileIndex === null ? "All Documents" : files[activeFileIndex]?.name}
                    </span>
                    <select 
                      value={activeFileIndex === null ? 'all' : activeFileIndex.toString()} 
                      onChange={(e) => setActiveFileIndex(e.target.value === 'all' ? null : Number(e.target.value))}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    >
                      <option value="all" className="bg-[#111318] text-white">All Documents</option>
                      {files.map((f, i) => (
                        <option key={i} value={i} className="bg-[#111318] text-white">
                          {f.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={14} className="text-[#6b7280] absolute right-1 z-0 group-hover:text-white pointer-events-none" />
                  </div>
                ) : (
                  <h2 className="text-sm font-bold text-white">
                    No Document Selected
                  </h2>
                )}
                <div className="flex items-center gap-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${activeFileIndex !== null ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
                  {/* <span className="text-[10px] font-mono text-[#6b7280] uppercase tracking-tighter">
                    {activeFileIndex !== null ? 'Gemini-2.0 Flash Active' : 'System Ready'}
                  </span> */}
                </div>
              </div>
           </div>
           
           {/* <div className="flex items-center gap-3">
              <button 
                type="button"
                className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg border border-white/5 text-[10px] font-bold hover:bg-white/10 transition-colors">
                <Zap size={12} className="text-[#2563eb]" /> 
                UPGRADE PRO
              </button>
              <button 
                type="button"
                className="p-2 bg-white/5 rounded-lg border border-white/5 text-[#6b7280] hover:text-white transition-colors">
                <Share2 size={16} />
              </button>
           </div> */}
        </header>

        {/* Chat Body */}
        <div className="flex-1 overflow-y-auto min-h-0 p-4 md:p-8 space-y-6">
           <div className="max-w-3xl mx-auto space-y-8">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex gap-4 ${msg.type === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 border border-white/5 ${msg.type === 'ai' ? 'bg-[#1a1d25]' : 'bg-[#2563eb]'}`}>
                    {msg.type === 'ai' ? <Cpu size={20} className="text-[#2563eb]" /> : <MessageSquare size={20} className="text-white" />}
                  </div>
                  <div className={`flex-1 flex flex-col ${msg.type === 'user' ? 'items-end' : ''}`}>
                    <div className={`p-4 rounded-2xl max-w-[85%] text-sm leading-relaxed border ${
                      msg.type === 'ai' 
                        ? 'bg-[#111318] border-white/5 text-[#e5e7eb] rounded-tl-none' 
                        : 'bg-[#2563eb] border-[#2563eb] text-white rounded-tr-none'
                    }`}>
                      {msg.type === 'ai' ? (
                        <AIMessageRender 
                          text={msg.text} 
                          isStreaming={msg.isStreaming}
                          onComplete={() => {
                            setMessages(msgs => msgs.map(m => m.id === msg.id ? { ...m, isStreaming: false } : m));
                          }}
                        />
                      ) : (
                        msg.text
                      )}
                      {msg.documentName && (
                        <div className={`mt-3 pt-2 border-t border-white/10 text-[9px] font-bold uppercase tracking-[0.1em] ${msg.type === 'ai' ? 'text-[#2563eb]' : 'text-blue-200 opacity-80'}`}>
                          Ref: {msg.documentName}
                        </div>
                      )}
                    </div>
                    <div className="flex flex-col gap-1 mt-2">
                       <span className={`text-[10px] text-[#6b7280] font-mono ${msg.type === 'user' ? 'text-right' : 'text-left'}`}>
                         {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                       </span>
                       <button
                         onClick={() => handleCopy(msg.text, msg.id)}
                         className={`flex items-center gap-1.5 text-[10px] text-[#6b7280] hover:text-white transition-colors ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                       >
                         {copiedId === msg.id ? <Check size={12} className="text-green-500" /> : <Copy size={12} />}
                         {copiedId === msg.id ? 'Copied' : 'Copy'}
                       </button>
                    </div>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex gap-4">
                  <div className="w-10 h-10 rounded-xl bg-[#1a1d25] border border-white/5 flex items-center justify-center">
                    <Cpu size={20} className="text-[#2563eb]" />
                  </div>
                  <div className="bg-[#111318] border border-white/5 px-6 py-4 rounded-2xl rounded-tl-none flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 bg-[#6b7280] rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1.5 h-1.5 bg-[#6b7280] rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-[#6b7280] rounded-full animate-bounce"></div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
           </div>
        </div>

        {/* Input Area */}
        <div className="flex-shrink-0 p-4 bg-[#0a0c10] border-t border-white/5 relative z-40">
          <div className="max-w-3xl mx-auto flex flex-col gap-3">
            
            <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-hide">
              {[
                  { id: 'QA', label: 'Semantic QA', icon: MessageSquare },
                  { id: 'Summary', label: 'Summary Engine', icon: FileText },
                  { id: 'Slides', label: 'Slide Generator', icon: Layout },
                  { id: 'Viz', label: 'Visualizer', icon: Share2 },
              ].map((item) => (
                <div key={item.id} className="relative flex items-center">
                  <button
                    type="button"
                    onClick={() => setMode(item.id as Mode)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold whitespace-nowrap transition-all ${mode === item.id ? 'bg-[#2563eb]/20 text-[#2563eb] border border-[#2563eb]/30 ring-1 ring-[#2563eb]/30' : 'bg-[#1a1d25] text-[#6b7280] hover:bg-white/10 hover:text-white border border-white/5'}`}
                  >
                    <item.icon size={14} className={mode === item.id ? "text-[#2563eb]" : "text-[#6b7280]"} />
                    {item.label}
                    {mode === 'Summary' && item.id === 'Summary' && (
                      <span className="flex items-center gap-1 border-l border-[#2563eb]/30 pl-2 ml-1 pointer-events-none">
                        {summarySubMode} <ChevronDown size={12} />
                      </span>
                    )}
                  </button>
                  {mode === 'Summary' && item.id === 'Summary' && (
                    <select
                      value={summarySubMode}
                      onChange={(e) => setSummarySubMode(e.target.value as SummarySubMode)}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    >
                      <option value="Snapshot" className="bg-[#111318] text-white">Snapshot</option>
                      <option value="Overview" className="bg-[#111318] text-white">Overview</option>
                      <option value="Deep Dive" className="bg-[#111318] text-white">Deep Dive</option>
                    </select>
                  )}
                </div>
              ))}
            </div>

            <div className="relative">
              <input 
                type="text" 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder={mode === 'Summary' ? `Explain with ${summarySubMode}...` : `Type your request for ${mode} mode...`}
                className="w-full bg-[#1a1d25] border border-white/5 rounded-2xl pl-12 pr-12 py-3.5 text-sm focus:outline-none focus:border-[#2563eb]/50 transition-all shadow-xl text-white"
              />
              <div 
                className="absolute left-2.5 top-1/2 -translate-y-1/2 p-2 hover:bg-white/5 rounded-lg text-[#6b7280] hover:text-white transition-colors cursor-pointer flex items-center justify-center"
                onClick={triggerFileUpload}
              >
                <Upload size={18} />
              </div>
              <button 
                type="button"
                onClick={handleSend}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 w-8 h-8 bg-[#2563eb] text-white rounded-lg flex items-center justify-center hover:scale-105 transition-all shadow-lg shadow-blue-900/20"
              >
                <Send size={14} />
              </button>
            </div>
            
            <div className="flex justify-between items-center px-1 mt-1">
              {/* <div className="flex gap-4">
                <span className="text-[10px] font-bold text-[#6b7280] uppercase tracking-widest flex items-center gap-1 focus-within:text-[#2563eb]">
                  <Zap size={10} /> Active Tokens: 1,402
                </span>
              </div> */}
              {/* <p className="text-[9px] text-[#6b7280] font-medium tracking-wide">AI Studio preview</p> */}
            </div>
          </div>
        </div>
      </main>
      
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-[#2563eb]/5 rounded-full blur-[120px] -z-10 pointer-events-none"></div>
    </motion.div>
  );
}