import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/features/auth/hooks/authHooks'
import { ROUTES } from '@/routes/paths'

interface ProtectedRouteProps {
  children: React.ReactNode
  requireAuth?: boolean
  requireAdmin?: boolean
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requireAuth = true,
  requireAdmin = false,
}) => {
  const { isAuthenticated, user } = useAuth()
  const location = useLocation()
  const isAdmin = user?.role?.toLowerCase() === 'admin'

  if (requireAuth && !isAuthenticated) {
    // Redirect unauthenticated user to login page
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  if (!requireAuth && isAuthenticated) {
    // Redirect authenticated user away from auth pages to chat
    return <Navigate to={ROUTES.chat} replace />
  }

  if (requireAdmin && !isAdmin) {
    // Redirect non-admin user away from admin-only pages to chat
    return <Navigate to={ROUTES.chat} replace />
  }

  return <>{children}</>
}
