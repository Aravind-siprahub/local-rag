import { createBrowserRouter } from 'react-router-dom'

import { AppLayout } from '@/layouts/AppLayout'
import {
  DashboardPage,
  DocumentsPage,
  NotFoundPage,
  ChatPage,
  UploadPage,
  SettingsPage,
} from '@/pages'
import { ROUTES } from '@/routes/paths'

export const appRouter = createBrowserRouter([
  {
    path: ROUTES.dashboard,
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <DashboardPage />,
      },
      {
        path: ROUTES.documents.slice(1),
        element: <DocumentsPage />,
      },
      {
        path: ROUTES.upload.slice(1),
        element: <UploadPage />,
      },
      {
        path: ROUTES.chat.slice(1),
        element: <ChatPage />,
      },
      {
        path: ROUTES.settings.slice(1),
        element: <SettingsPage />,
      },
      {
        path: '*',
        element: <NotFoundPage />,
      },
    ],
  },
])

