import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "MediNova X research console",
  description: "U-MERGE image-report correspondence. Research and education only.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="banner">Research prototype. Not for clinical use; output is not a diagnosis.</div>
        <main>{children}</main>
      </body>
    </html>
  );
}
