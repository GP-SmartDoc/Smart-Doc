import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { useAuth } from '../lib/AuthContext';
import { LogIn } from 'lucide-react';

export default function LandingPage() {
  const [activeReveal, setActiveReveal] = useState<{ [key: string]: boolean }>({});
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const { user, signInWithGoogle, signInWithGithub, signInAsGuest, signInWithEmail, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [showEmailForm, setShowEmailForm] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [authError, setAuthError] = useState('');

  useEffect(() => {
    // Hide loading screen very quickly now
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Check if redirecting from a protected route
    if (location.search.includes('login=true')) {
      setShowLoginModal(true);
    }
  }, [location]);

  const handleAuthAction = async () => {
    if (user) {
      navigate('/chat');
    } else {
      setShowLoginModal(true);
      setShowEmailForm(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      await signInWithGoogle();
      setShowLoginModal(false);
      navigate('/chat');
    } catch (error) {
      console.error("Login failed:", error);
      setAuthError('Google login failed.');
    }
  };

  const handleGithubLogin = async () => {
    try {
      await signInWithGithub();
      setShowLoginModal(false);
      navigate('/chat');
    } catch (error) {
      console.error("Github login failed:", error);
      setAuthError('Github login failed.');
    }
  };

  const handleGuestLogin = async () => {
    try {
      await signInAsGuest();
      setShowLoginModal(false);
      navigate('/chat');
    } catch (error) {
      console.error("Guest login failed:", error);
      setAuthError('Guest login failed.');
    }
  };

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError('');
    try {
      await signInWithEmail(email, password, isSignUp);
      setShowLoginModal(false);
      navigate('/chat');
    } catch (error: any) {
      console.error("Email auth failed:", error);
      setAuthError(error.message || 'Authentication failed.');
    }
  };

  return (
    <AnimatePresence mode="wait">
      {isLoading ? (
        <motion.div 
          key="loader"
          exit={{ opacity: 0, scale: 1.05, filter: 'blur(10px)' }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          className="fixed inset-0 z-[9999] bg-[#0a0c10] flex items-center justify-center overflow-hidden"
        >
          {/* Atmospheric background */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60vw] h-[60vw] bg-[radial-gradient(circle,rgba(37,99,235,0.15)_0%,transparent_60%)] opacity-50 blur-3xl rounded-full mix-blend-screen"></div>
          
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="relative flex items-center text-7xl md:text-[140px] font-black tracking-tighter"
          >
            <span className="text-white z-10 drop-shadow-[0_0_15px_rgba(255,255,255,0.4)]">S</span>
            <motion.span 
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: "auto", opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden inline-block text-white z-0"
            >
              <span>mart</span>
            </motion.span>
            
            <span className="text-[#2563eb] z-10 drop-shadow-[0_0_30px_rgba(37,99,235,0.5)] leading-none text-8xl md:text-[150px] relative -top-1 md:-top-2 lg:-top-3 ml-1 md:ml-2">D</span>
            <motion.span 
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: "auto", opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden inline-block text-[#2563eb] z-0"
            >
              <span>oc</span>
            </motion.span>
          </motion.div>
        </motion.div>
      ) : (
        <motion.div 
          key="content"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="bg-[#0f1117] text-[#e5e7eb] font-sans overflow-x-hidden min-h-screen"
        >
      {/* Navigation */}
      <nav className="fixed top-0 left-0 w-full h-[70px] bg-[#0f1117]/80 backdrop-blur-md z-[1000] border-b border-white/5">
        <div className="max-w-7xl mx-auto px-8 h-full flex items-center justify-between">
          <Link to="/" className="text-2xl font-extrabold text-white tracking-tighter flex items-center">
            S<span className="tracking-tight mr-[1px]">mart</span><span className="text-[#2563eb]">D</span><span className="text-[#2563eb] tracking-tight">oc</span>
          </Link>
          <div className="hidden md:flex items-center gap-8">
            <a href="#problem" className="text-sm font-medium text-[#6b7280] hover:text-white transition-colors">Problem</a>
            <a href="#solution" className="text-sm font-medium text-[#6b7280] hover:text-white transition-colors">Solution</a>
            <a href="#features" className="text-sm font-medium text-[#6b7280] hover:text-white transition-colors">Features</a>
            <a href="#how-it-works" className="text-sm font-medium text-[#6b7280] hover:text-white transition-colors">How It Works</a>
            <button 
              onClick={handleAuthAction}
              disabled={loading}
              className="px-5 py-2 bg-[#2563eb] text-white rounded-full font-semibold text-sm hover:scale-105 transition-all shadow-[0_4px_20px_rgba(37,99,235,0.2)] disabled:opacity-50"
            >
              {user ? 'Go to App' : 'Try SmartDoc →'}
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <header className="pt-[120px] md:pt-[180px] pb-[60px] md:pb-[100px] text-center relative overflow-hidden">
        <div className="absolute top-[-20%] left-1/2 -translate-x-1/2 w-[1000px] h-[1000px] bg-[radial-gradient(circle,rgba(37,99,235,0.1)_0%,transparent_70%)] -z-10 opacity-50"></div>
        <div className="container mx-auto px-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
          <h1 className="text-5xl md:text-7xl font-extrabold leading-[1.1] mb-6 tracking-tighter text-white">
            Your Documents,<br />Finally <span className="text-[#2563eb]">Intelligent</span>
          </h1>
          <p className="text-lg md:text-xl text-[#6b7280] max-w-3xl mx-auto mb-12 leading-relaxed">
            Upload any PDF and ask questions, get summaries, generate slides, and visualize insights — all through one AI-powered chat interface.
          </p>
          <div className="flex flex-wrap justify-center gap-4 mb-20">
            <button 
              onClick={handleAuthAction}
              disabled={loading}
              className="px-8 py-4 bg-[#2563eb] text-white rounded-xl font-bold text-lg hover:-translate-y-1 transition-all shadow-[0_10px_25px_rgba(37,99,235,0.2)] disabled:opacity-50"
            >
              {user ? 'Open Dashboard →' : 'Try SmartDoc →'}
            </button>
            <button className="px-8 py-4 border border-white/10 text-white rounded-xl font-bold text-lg hover:bg-white/5 transition-all">
              See How It Works
            </button>
          </div>
          
          {/* Chat Mockup */}
          <div className="bg-[#1a1d25]/60 backdrop-blur-3xl rounded-[24px] border border-white/10 max-w-[900px] mx-auto h-[400px] md:h-[500px] relative shadow-[0_40px_100px_-20px_rgba(37,99,235,0.2)] flex flex-col overflow-hidden">
            <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-[#ffbd2e]"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-[#28c940]"></div>
                <span className="text-xs text-[#6b7280] ml-4 bg-white/5 px-3 py-1 rounded">Project_Quarterly_Results_2025.pdf</span>
              </div>
              <div className="bg-[#2563eb]/10 text-[#2563eb] px-3 py-1 rounded-full text-[10px] font-bold border border-[#2563eb]/20 uppercase">
                QA MODE
              </div>
            </div>
            <div className="flex-1 p-8 flex flex-col gap-6 overflow-y-auto">
              <div className="self-start bg-[#111318] text-[#e5e7eb] px-5 py-4 rounded-[18px] rounded-bl-[4px] border border-white/5 max-w-[80%] text-sm text-left">
                Hello! I've indexed your document. How can I help you today?
              </div>
              <div className="self-end bg-[#2563eb] text-white px-5 py-4 rounded-[18px] rounded-br-[4px] max-w-[80%] text-sm text-left">
                Can you summarize the main risks mentioned in section 4?
              </div>
              <div className="self-start bg-[#111318] text-[#e5e7eb] px-5 py-4 rounded-[18px] rounded-bl-[4px] border border-white/5 max-w-[80%] text-sm text-left">
                Based on Section 4 (Risk Assessment), the primary risks are: 1. Supply chain delays in Southeast Asia, 2. Currency volatility, and 3. Potential regulatory changes in data privacy. Would you like a deeper dive into any of these?
              </div>
            </div>
            <div className="p-6 border-t border-white/5 flex gap-4">
              <div className="flex-1 bg-[#0f1117] border border-white/10 rounded-xl h-11"></div>
              <div className="bg-[#2563eb] w-11 h-11 rounded-xl flex items-center justify-center text-white font-bold">➔</div>
            </div>
          </div>
        </div>
      </header>

      {/* Problem Section */}
      <section id="problem" className="py-16 md:py-24 animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="container mx-auto px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-extrabold mb-4 text-white">Reading docs shouldn't be this painful</h2>
            <p className="text-[#6b7280] text-lg max-w-2xl mx-auto">Modern professionals waste hours every week digging through static PDF files.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { icon: '📄', title: 'Too much to read', desc: 'Long reports take hours to digest. Most of the content isn\'t even relevant to your specific needs.' },
              { icon: '🔍', title: 'Can\'t find what you need', desc: 'CTRL+F is not intelligence. It just finds keywords, it doesn\'t give you conceptual answers.' },
              { icon: '🧩', title: 'No structure, no insight', desc: 'Raw PDFs are just text on pages. They don\'t tell you what matters most or how concepts connect.' }
            ].map((item, i) => (
              <div key={i} className="bg-[#111318] p-10 rounded-[24px] border border-white/5 hover:border-[#2563eb] hover:shadow-[0_10px_40px_-10px_rgba(37,99,235,0.2)] hover:-translate-y-1 transition-all group">
                <div className="text-4xl mb-6">{item.icon}</div>
                <h3 className="text-xl font-bold mb-4 text-white">{item.title}</h3>
                <p className="text-[#6b7280]">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section id="solution" className="py-16 md:py-24 bg-[#111318] animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="container mx-auto px-8 text-center">
          <div className="mb-16">
            <h2 className="text-4xl font-extrabold mb-4 text-white">SmartDoc fixes all of that</h2>
            <p className="text-[#6b7280] text-lg max-w-2xl mx-auto">One upload. Four powerful AI modes. Every document becomes a conversation.</p>
          </div>
          <div className="flex flex-wrap justify-center gap-4 mb-20">
            {['🔍 QA', '📝 Summary', '🎞️ Slide Gen', '🗺️ Visualization'].map((mode, i) => (
              <div key={i} className="bg-[#2563eb]/10 text-[#2563eb] px-6 py-3 rounded-full font-bold border border-[#2563eb]/20 flex items-center gap-2">
                {mode}
              </div>
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
            {[
              { title: 'Instant QA', desc: 'Ask natural language questions and get grounded answers cited directly from the document source.' },
              { title: 'Multi-Depth Summary', desc: 'Choose your detail level: a 1-sentence Snapshot, a balanced Overview, or a meticulous Deep Dive.' },
              { title: 'Outline to Presentation', desc: 'Instantly extract key arguments and data points into a structured outline ready for your next slide deck.' },
              { title: 'Knowledge Mapping', desc: 'Visualize complex relationships and workflows within the text through auto-generated Mermaid diagrams.' }
            ].map((item, i) => (
              <div key={i}>
                <h4 className="text-lg font-bold mb-4 text-white">{item.title}</h4>
                <p className="text-[#6b7280] text-sm leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-16 md:py-24 animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="container mx-auto px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-extrabold text-white">Everything you need. Nothing you don't.</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { title: '📤 Multi-PDF Upload', desc: 'Upload multiple documents at once and switch context mid-conversation without missing a beat.' },
              { title: '🤖 AI-Powered QA', desc: 'Advanced RAG pipeline ensures your answers are accurate and never hallucinated.' },
              { title: '📝 Smart Summaries', desc: 'Summarize thousands of pages in seconds with context-aware length control.' },
              { title: '🎞️ Slide Generator', desc: 'Convert dense whitepapers into bite-sized presentation structures automatically.' },
              { title: '🗺️ Mermaid Visualizations', desc: 'Extract flowcharts and sequence diagrams to see how your project actually works.' },
              { title: '⚡ Typewriter Responses', desc: 'Fluid, real-time response streaming that makes the AI feel like a true teammate.' }
            ].map((item, i) => (
              <div key={i} className="bg-[#1a1d25] p-8 rounded-[20px] border border-white/5 hover:-translate-y-2 hover:bg-[#1e222b] transition-all">
                <h3 className="text-lg font-bold mb-3 text-white">{item.title}</h3>
                <p className="text-[#6b7280] text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Section (Fake Interactive) */}
      <section id="demo" className="py-16 md:py-24 bg-[#0a0c10] animate-in fade-in slide-in-from-bottom-8 duration-700">
         <div className="container mx-auto px-8">
           <div className="text-center mb-16">
             <h2 className="text-4xl font-extrabold mb-4 text-white">See it in action</h2>
             <p className="text-[#6b7280] text-lg">Our AI analyzes and responds to complex document queries in real-time.</p>
           </div>
           
           <div className="bg-[#1a1d25]/60 backdrop-blur-3xl rounded-[24px] border border-white/10 max-w-[800px] mx-auto h-[350px] md:h-[450px] flex flex-col overflow-hidden shadow-2xl">
              <div className="px-6 py-4 border-b border-white/5">
                <span className="font-bold text-white">SmartDoc Demo</span>
              </div>
              <div className="flex-1 p-6 flex flex-col gap-4 bg-[#0a0c10]/50 overflow-y-auto">
                 <div className="self-end bg-[#2563eb] text-white px-4 py-2 rounded-2xl rounded-br-none text-sm max-w-[80%] opacity-0 animate-[fadeIn_0.5s_forwards_1s]">
                    Summarize the risk distribution?
                 </div>
                 <div className="self-start bg-[#111318] border border-white/5 text-[#e5e7eb] px-4 py-2 rounded-2xl rounded-bl-none text-sm max-w-[80%] opacity-0 animate-[fadeIn_0.5s_forwards_3s]">
                    The primary risks are concentrated in supply chain (45%) and market volatility (30%).
                 </div>
              </div>
              <div className="p-4 bg-[#111318]/50 border-t border-white/5">
                <button 
                  onClick={handleAuthAction}
                  className="w-full py-3 bg-[#2563eb] text-white rounded-lg font-bold flex items-center justify-center gap-2 hover:bg-blue-700 transition-colors"
                >
                  Try the real thing →
                </button>
              </div>
           </div>
         </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-16 md:py-24 animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="container mx-auto px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-extrabold text-white">Three steps to document intelligence</h2>
          </div>
          <div className="flex flex-col md:flex-row justify-between items-start gap-12 relative">
            {[
              { num: '1', icon: '📤', title: 'Upload your PDF', desc: 'Drag and drop any document. Multiple files supported simultaneously.' },
              { num: '2', icon: '💬', title: 'Choose your mode', desc: 'Switch between QA, Summary, Slides, or Visualization modes anytime.' },
              { num: '3', icon: '🚀', title: 'Get your answer', desc: 'SmartDoc reads, reasons, and responds in seconds with perfect accuracy.' }
            ].map((step, i) => (
              <div key={i} className="flex-1 text-center relative z-10">
                <div className="w-12 h-12 bg-[#2563eb] text-white rounded-full flex items-center justify-center font-black text-xl mx-auto mb-6 shadow-[0_0_20px_rgba(37,99,235,0.4)]">
                  {step.num}
                </div>
                <div className="text-4xl mb-4">{step.icon}</div>
                <h3 className="text-xl font-bold mb-2 text-white">{step.title}</h3>
                <p className="text-[#6b7280] text-sm">{step.desc}</p>
              </div>
            ))}
            {/* Connecting line for desktop */}
            <div className="hidden md:block absolute top-6 left-[15%] right-[15%] h-0.5 bg-gradient-to-r from-[#2563eb] to-transparent z-0"></div>
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-16 md:py-24 text-center animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="container mx-auto px-8">
          <h2 className="text-3xl font-extrabold mb-12 text-white">Built with serious technology</h2>
          <div className="flex flex-wrap justify-center gap-3 mb-8">
            {['Python', 'Flask', 'Lang Chain', 'Lang Graph', 'Mermaid.js', 'RAG Pipeline', 'JavaScript', 'Tailwind CSS'].map((tech, i) => (
              <div key={i} className="bg-[#1a1d25] px-5 py-2 rounded-lg font-bold text-sm text-white border border-white/5">
                {tech}
              </div>
            ))}
          </div>
          <p className="text-[#6b7280]">SmartDoc combines retrieval-augmented generation with large language models to deliver grounded, accurate answers — not hallucinations.</p>
        </div>
      </section>

      {/* CTA Final */}
      <section className="py-20 md:py-32 bg-gradient-to-b from-[#0f1117] to-[#1a1d25] text-center animate-in fade-in slide-in-from-bottom-8 duration-700">
        <div className="container mx-auto px-8">
          <h2 className="text-5xl font-extrabold mb-6 text-white">Ready to talk to your documents?</h2>
          <p className="text-xl text-[#6b7280] mb-12">Built as a graduation project. Designed like a product.</p>
          <div className="flex flex-wrap justify-center gap-4">
            <button 
              onClick={handleAuthAction}
              disabled={loading}
              className="px-10 py-5 bg-[#2563eb] text-white rounded-xl font-bold text-xl hover:scale-105 transition-all shadow-xl shadow-blue-900/10 disabled:opacity-50"
            >
              {user ? 'Open Dashboard →' : 'Launch SmartDoc →'}
            </button>
            <a href="https://github.com/GP-SmartDoc/Smart-Doc" target="_blank" rel="noopener noreferrer" className="px-10 py-5 border border-white/10 text-white rounded-xl font-bold text-xl hover:bg-white/5 transition-all">
              View on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* Login Modal */}
      <AnimatePresence>
        {showLoginModal && (
          <div className="fixed inset-0 z-[2000] flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowLoginModal(false)}
              className="absolute inset-0 bg-[#000]/60 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative bg-[#1a1d25] border border-white/10 rounded-[32px] w-full max-w-md p-10 shadow-2xl overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-full h-1 bg-[#2563eb]" />
              <button 
                onClick={() => setShowLoginModal(false)}
                className="absolute top-6 right-6 text-[#6b7280] hover:text-white transition-colors"
              >
                ✕
              </button>
              
              <div className="text-center mb-10">
                <div className="w-16 h-16 bg-[#2563eb]/10 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <LogIn size={32} className="text-[#2563eb]" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-2">Welcome to SmartDoc</h3>
                <p className="text-[#6b7280]">Connect your account to start analyzing your documents.</p>
              </div>

              <div className="space-y-4">
                {authError && <p className="text-red-500 text-xs text-center">{authError}</p>}

                {showEmailForm ? (
                  <form onSubmit={handleEmailSubmit} className="space-y-3">
                    <input 
                      type="email" 
                      required
                      placeholder="Email Address" 
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full h-12 bg-[#0f1117] border border-white/10 rounded-xl px-4 text-sm text-white focus:outline-none focus:border-[#2563eb]"
                    />
                    <input 
                      type="password" 
                      required
                      placeholder="Password" 
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="w-full h-12 bg-[#0f1117] border border-white/10 rounded-xl px-4 text-sm text-white focus:outline-none focus:border-[#2563eb]"
                    />
                    <button 
                      type="submit"
                      className="w-full h-12 bg-[#2563eb] text-white rounded-xl font-bold flex items-center justify-center hover:bg-blue-600 transition-colors"
                    >
                      {isSignUp ? 'Sign Up' : 'Sign In'}
                    </button>
                    <p className="text-xs text-center text-[#6b7280]">
                      {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
                      <button type="button" className="text-[#2563eb]" onClick={() => setIsSignUp(!isSignUp)}>
                        {isSignUp ? 'Sign In' : 'Sign Up'}
                      </button>
                    </p>
                    <button type="button" onClick={() => setShowEmailForm(false)} className="w-full text-xs text-center text-[#6b7280] hover:text-white mt-2">
                       ← Back to options
                    </button>
                  </form>
                ) : (
                  <>
                    <button 
                      onClick={handleGoogleLogin}
                      className="w-full h-14 bg-white text-black rounded-xl font-bold flex items-center justify-center gap-3 hover:bg-white/90 transition-colors"
                    >
                      <img src="https://www.google.com/favicon.ico" alt="Google" className="w-5 h-5" />
                      Continue with Google
                    </button>
                    
                    <button 
                      onClick={handleGithubLogin}
                      className="w-full h-14 bg-[#24292e] text-white rounded-xl font-bold flex items-center justify-center gap-3 hover:bg-[#2c3137] transition-colors"
                    >
                       {/* Simple generic SVG for Github since lucide might not have exact match without import */}
                      <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                      </svg>
                      Continue with GitHub
                    </button>

                    <div className="relative flex items-center py-2">
                      <div className="flex-grow border-t border-white/5"></div>
                      <span className="flex-shrink mx-4 text-xs font-bold text-[#6b7280] uppercase tracking-widest">Or</span>
                      <div className="flex-grow border-t border-white/5"></div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <button onClick={() => setShowEmailForm(true)} className="h-12 border border-white/5 rounded-xl text-sm font-bold text-white hover:bg-white/5 transition-colors">Email</button>
                      <button onClick={handleGuestLogin} className="h-12 border border-white/5 rounded-xl text-sm font-bold text-white hover:bg-white/5 transition-colors">Guest</button>
                    </div>
                  </>
                )}
              </div>

              <p className="text-center text-[10px] text-[#6b7280] mt-10 leading-relaxed">
                By signing up, you agree to our <span className="text-white underline cursor-pointer">Terms of Service</span> and <span className="text-white underline cursor-pointer">Privacy Policy</span>.
              </p>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <footer className="py-10 border-t border-white/5 text-center text-[#6b7280] text-sm">
        <div className="container mx-auto px-8">
          SmartDoc &copy; 2025 &middot; Graduation Project &middot; Built with &hearts;
        </div>
      </footer>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
