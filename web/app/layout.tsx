import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { Navigation } from '@/components/Navigation';
import { ScheduleInfo } from '@/components/ScheduleInfo';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Stock Prediction Dashboard',
  description: 'MLOps dashboard for stock movement predictions',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen flex flex-col">
            <Navigation />
            <main className="container mx-auto px-4 py-8 flex-1">
              {children}
            </main>
            <ScheduleInfo />
          </div>
        </Providers>
      </body>
    </html>
  );
}
