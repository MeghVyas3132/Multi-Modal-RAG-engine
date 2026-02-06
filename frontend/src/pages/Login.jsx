import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mail, Lock, ArrowRight } from 'lucide-react';

const Login = () => {
    const navigate = useNavigate();
    const [isSignup, setIsSignup] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleAuth = (e) => {
        e.preventDefault();
        if (!email || !password) return;

        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('userEmail', email);
        navigate('/chat');
    };

    const handleGoogleLogin = () => {
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('userEmail', 'user@gmail.com');
        navigate('/chat');
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-50 font-sans">
            <div className="w-full max-w-[400px] bg-white rounded-2xl shadow-xl shadow-gray-200/50 overflow-hidden transition-all duration-300 border border-gray-100">
                <div className="p-8">
                    <div className="text-center mb-8">
                        <div className="flex justify-center mb-5">
                            <div className="w-10 h-10 bg-black rounded-lg flex items-center justify-center shadow-lg shadow-black/10">
                                <span className="text-xl font-bold text-white">AI</span>
                            </div>
                        </div>
                        <h2 className="text-2xl font-bold text-gray-900 tracking-tight">
                            {isSignup ? 'Create account' : 'Welcome back'}
                        </h2>
                        <p className="mt-2 text-sm text-gray-500">
                            {isSignup ? 'Start your journey today.' : 'Enter your details to sign in.'}
                        </p>
                    </div>

                    <form onSubmit={handleAuth} className="space-y-4">
                        <div>
                            <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wider">Email</label>
                            <div className="relative">
                                <span className="absolute left-3 top-2.5 text-gray-400">
                                    <Mail className="w-4 h-4" />
                                </span>
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="name@example.com"
                                    className="w-full pl-9 pr-4 py-2 bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-black/5 focus:border-gray-400 transition-all text-sm text-gray-800 placeholder-gray-400"
                                    required
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wider">Password</label>
                            <div className="relative">
                                <span className="absolute left-3 top-2.5 text-gray-400">
                                    <Lock className="w-4 h-4" />
                                </span>
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className="w-full pl-9 pr-4 py-2 bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-black/5 focus:border-gray-400 transition-all text-sm text-gray-800 placeholder-gray-400"
                                    required
                                />
                            </div>
                        </div>

                        <button
                            type="submit"
                            className="w-full flex items-center justify-center py-2.5 px-4 bg-gray-900 hover:bg-black text-white rounded-lg font-medium shadow-sm hover:shadow-md transition-all duration-200 transform active:scale-[0.98]"
                        >
                            {isSignup ? 'Sign up' : 'Sign in'}
                            <ArrowRight className="w-4 h-4 ml-2" />
                        </button>
                    </form>

                    <div className="relative my-6">
                        <div className="absolute inset-0 flex items-center">
                            <div className="w-full border-t border-gray-200"></div>
                        </div>
                        <div className="relative flex justify-center text-xs uppercase">
                            <span className="px-2 bg-white text-gray-400 font-medium tracking-wider">Or continue with</span>
                        </div>
                    </div>

                    <button
                        onClick={handleGoogleLogin}
                        className="w-full flex items-center justify-center py-2.5 px-4 border border-gray-200 rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-gray-300 transition-all duration-200 shadow-sm"
                    >
                        <svg className="w-5 h-5 mr-3" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                        </svg>
                        <span className="font-medium text-sm">Google</span>
                    </button>

                    <div className="mt-8 text-center bg-gray-50 -mx-8 -mb-8 py-4 border-t border-gray-100">
                        <p className="text-sm text-gray-500">
                            {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
                            <button
                                onClick={() => setIsSignup(!isSignup)}
                                className="font-semibold text-gray-900 hover:underline transition-colors block w-full mt-1"
                            >
                                {isSignup ? 'Log in' : 'Create account'}
                            </button>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
