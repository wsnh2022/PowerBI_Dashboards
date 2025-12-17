# Power BI Desktop (Microsoft Store) vs EXE Installation

## Clean Uninstall, Reinstallation, and MySQL Connectivity — Detailed Reference Document

---

## 1. Problem Context

Power BI Desktop installed via the **Microsoft Store** behaves differently from a traditional Windows desktop application. When facing persistent issues such as:

* MySQL authentication failures
* Incorrect Windows authentication prompts
* Connector handshake errors
* Power BI not appearing in Revo Uninstaller or Control Panel

A **clean uninstall and reinstall using the EXE version** is often required.

This document explains **why this happens**, **how to remove the Store version correctly**, and **how to reinstall Power BI cleanly for stable MySQL connectivity**.

---

## 2. Why Power BI Desktop Does Not Appear in Revo or Control Panel

### Root Reason

Power BI Desktop installed from the Microsoft Store is a **UWP / MSIX package**, not a traditional Win32 application.

### Consequences

* No MSI / EXE installer
* No Control Panel entry
* No standard uninstall registry keys
* Invisible to tools like Revo Uninstaller

### Where Store Apps Live

```
C:\Program Files\WindowsApps\
```

This directory is protected by Windows and hidden by default.

### Expected Behavior

* ❌ Not visible in Control Panel
* ❌ Not visible in Revo Uninstaller
* ✅ Visible only in Microsoft Store (Open / Uninstall button)

This is **normal behavior**, not a system fault.

---

## 3. Correct Way to Uninstall Microsoft Store Version of Power BI

### Method A — Microsoft Store (Recommended)

1. Open **Microsoft Store**
2. Go to **Library**
3. Locate **Power BI Desktop**
4. Click **Uninstall**

This cleanly removes the MSIX package.

---

### Method B — PowerShell Forced Removal (If Store Fails)

Run **PowerShell as Administrator**:

```powershell
Get-AppxPackage *PowerBI* | Remove-AppxPackage
```

This force-removes the Store-installed Power BI Desktop.

---

## 4. Manual Cleanup After Uninstall (Critical)

Even after uninstalling, Power BI leaves behind **local data, cached credentials, and connector state**.

Delete the following folders if they exist:

```
C:\Users\<username>\Documents\Power BI Desktop
C:\Users\<username>\AppData\Local\Microsoft\Power BI
C:\Users\<username>\AppData\Local\Microsoft\Power BI Desktop
C:\Users\<username>\AppData\Roaming\Microsoft\Power BI
```

---

## 5. Clear Cached Credentials (Non‑Optional)

Power BI aggressively caches failed authentication attempts.

1. Open **Control Panel**
2. Go to **Credential Manager**
3. Open **Windows Credentials**
4. Delete any entries related to:

   * Power BI
   * MySQL
   * localhost
   * ODBC
   * SQL

Failure to do this can cause repeated authentication errors even after reinstall.

---

## 6. Reboot the System

Rebooting clears:

* Loaded connector DLLs
* Locked credential providers
* Residual MSIX bindings

Do not skip this step.

---

## 7. Correct Reinstallation Procedure (EXE Version)

### Why EXE Version Is Preferred

The EXE installer:

* Uses classic Win32 architecture
* Is visible to Control Panel and Revo
* Loads connectors reliably
* Avoids Store sandbox limitations
* Is more stable for MySQL connectivity

### Installation Steps

1. Download **Power BI Desktop (x64 EXE)** from Microsoft official website
2. Install normally

---

## 8. Install MySQL Connector (Mandatory)

Power BI **does not use MySQL Workbench**.

It requires:

* **MySQL Connector/NET**
* **64‑bit version only**
* Version **8.0.x**

Install the connector **after** installing Power BI Desktop.

Reboot again after installation.

---

## 9. MySQL Authentication Configuration

Power BI often fails with MySQL’s default authentication plugin.

Force classic authentication.

Run in MySQL Workbench:

```sql
ALTER USER 'root'@'localhost'
IDENTIFIED WITH mysql_native_password
BY 'your_password';

FLUSH PRIVILEGES;
```

Restart MySQL service.

---

## 10. First Connection Setup in Power BI

Use this exact sequence:

```
Home → Get Data → MySQL database
```

Connection details:

```
Server: localhost
Database: sakila
```

Authentication:

* Authentication kind: Database
* Select: Use alternate credentials
* Username: MySQL username
* Password: MySQL password
* Apply level: localhost:3306

Connect.

---

## 11. Common Errors and Their Meaning

### Windows Authentication Prompt

* Misleading UI text
* MySQL does NOT support Windows authentication
* Always use Database credentials

### Pre-login Handshake Error

* SQL Server connector used by mistake
* Wrong protocol
* Use MySQL database connector only

### Credential Loop

* Cached failed credentials
* Fix by clearing Credential Manager

---

## 12. Final Outcome

After following this document:

* Power BI loads MySQL tables immediately
* No Windows authentication prompts
* No handshake or TCP errors
* Stable, repeatable MySQL connections

---

## 13. Key Takeaway

This issue is caused by:

* Microsoft Store sandboxing
* Power BI credential cache behavior
* MySQL authentication plugin mismatch

A **clean EXE-based install + connector reset** is the correct engineering solution.

---

End of document.
