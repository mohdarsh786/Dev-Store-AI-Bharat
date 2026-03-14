"use client";

import React, { useState } from 'react';
import LoginModal from './LoginModal';
import SignupModal from './SignupModal';
import ForgotPasswordModal from './ForgotPasswordModal';

type AuthView = 'login' | 'signup' | 'forgot-password' | null;

interface AuthManagerProps {
  initialView?: AuthView;
  onClose?: () => void;
}

export default function AuthManager({ initialView = null, onClose }: AuthManagerProps) {
  const [currentView, setCurrentView] = useState<AuthView>(initialView);

  const handleClose = () => {
    setCurrentView(null);
    onClose?.();
  };

  return (
    <>
      <LoginModal
        isOpen={currentView === 'login'}
        onClose={handleClose}
        onSwitchToSignup={() => setCurrentView('signup')}
        onSwitchToForgotPassword={() => setCurrentView('forgot-password')}
      />

      <SignupModal
        isOpen={currentView === 'signup'}
        onClose={handleClose}
        onSwitchToLogin={() => setCurrentView('login')}
      />

      <ForgotPasswordModal
        isOpen={currentView === 'forgot-password'}
        onClose={handleClose}
        onSwitchToLogin={() => setCurrentView('login')}
      />
    </>
  );
}

// Hook to use auth manager
export function useAuthManager() {
  const [view, setView] = useState<AuthView>(null);

  return {
    showLogin: () => setView('login'),
    showSignup: () => setView('signup'),
    showForgotPassword: () => setView('forgot-password'),
    close: () => setView(null),
    AuthManager: () => <AuthManager initialView={view} onClose={() => setView(null)} />,
  };
}
