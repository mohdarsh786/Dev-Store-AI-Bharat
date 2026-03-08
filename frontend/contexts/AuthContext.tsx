"use client";

import React, { createContext, useContext, useState, useEffect } from 'react';

interface User {
  id: string;
  email: string;
  name: string;
  auth_provider: string;
  is_verified: boolean;
  avatar_url?: string;
  preferred_language: string;
  created_at: string;
  last_login_at?: string;
}

interface AuthContextType {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  loginWithGoogle: (googleData: any) => Promise<void>;
  loginWithGithub: (githubData: any) => Promise<void>;
  logout: () => void;
  forgotPassword: (email: string) => Promise<void>;
  resetPassword: (token: string, newPassword: string) => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

  // Load tokens from localStorage on mount
  useEffect(() => {
    const storedAccessToken = localStorage.getItem('access_token');
    const storedRefreshToken = localStorage.getItem('refresh_token');
    const storedUser = localStorage.getItem('user');

    if (storedAccessToken && storedUser) {
      setAccessToken(storedAccessToken);
      setRefreshToken(storedRefreshToken);
      setUser(JSON.parse(storedUser));
    }
    setIsLoading(false);
  }, []);

  // Save tokens to localStorage
  const saveTokens = (access: string, refresh: string, userData: User) => {
    localStorage.setItem('access_token', access);
    localStorage.setItem('refresh_token', refresh);
    localStorage.setItem('user', JSON.stringify(userData));
    setAccessToken(access);
    setRefreshToken(refresh);
    setUser(userData);
  };

  // Clear tokens
  const clearTokens = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    setAccessToken(null);
    setRefreshToken(null);
    setUser(null);
  };

  // Manual Login
  const login = async (email: string, password: string) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }

      const data = await response.json();
      saveTokens(data.access_token, data.refresh_token, data.user);
    } catch (error: any) {
      throw new Error(error.message || 'Login failed');
    }
  };

  // Manual Signup
  const signup = async (name: string, email: string, password: string) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Signup failed');
      }

      const data = await response.json();
      saveTokens(data.access_token, data.refresh_token, data.user);
    } catch (error: any) {
      throw new Error(error.message || 'Signup failed');
    }
  };

  // Google OAuth
  const loginWithGoogle = async (googleData: any) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/oauth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: googleData.email,
          name: googleData.name,
          auth_provider: 'google',
          oauth_provider_id: googleData.sub,
          avatar_url: googleData.picture,
        }),
      });

      if (!response.ok) {
        throw new Error('Google login failed');
      }

      const data = await response.json();
      saveTokens(data.access_token, data.refresh_token, data.user);
    } catch (error: any) {
      throw new Error(error.message || 'Google login failed');
    }
  };

  // GitHub OAuth
  const loginWithGithub = async (githubData: any) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/oauth/github`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: githubData.email,
          name: githubData.name,
          auth_provider: 'github',
          oauth_provider_id: githubData.id.toString(),
          avatar_url: githubData.avatar_url,
        }),
      });

      if (!response.ok) {
        throw new Error('GitHub login failed');
      }

      const data = await response.json();
      saveTokens(data.access_token, data.refresh_token, data.user);
    } catch (error: any) {
      throw new Error(error.message || 'GitHub login failed');
    }
  };

  // Logout
  const logout = () => {
    clearTokens();
  };

  // Forgot Password
  const forgotPassword = async (email: string) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/password/forgot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      if (!response.ok) {
        throw new Error('Failed to send reset email');
      }
    } catch (error: any) {
      throw new Error(error.message || 'Failed to send reset email');
    }
  };

  // Reset Password
  const resetPassword = async (token: string, newPassword: string) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/password/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, new_password: newPassword }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Password reset failed');
      }
    } catch (error: any) {
      throw new Error(error.message || 'Password reset failed');
    }
  };

  // Change Password
  const changePassword = async (currentPassword: string, newPassword: string) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/password/change`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Password change failed');
      }
    } catch (error: any) {
      throw new Error(error.message || 'Password change failed');
    }
  };

  const value = {
    user,
    accessToken,
    refreshToken,
    isLoading,
    isAuthenticated: !!user,
    login,
    signup,
    loginWithGoogle,
    loginWithGithub,
    logout,
    forgotPassword,
    resetPassword,
    changePassword,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
