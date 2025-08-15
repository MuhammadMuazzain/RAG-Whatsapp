#!/usr/bin/env python
"""
Document Management Utility for RAG Knowledge Base
View, backup, and manage documents in the vector store
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse

class DocumentManager:
    def __init__(self):
        self.vector_store_path = Path("vector_store")
        self.backup_path = Path("vector_store_backups")
        self.metadata_file = self.vector_store_path / "chunks.json"
        self.index_file = self.vector_store_path / "faiss.index"
    
    def list_documents(self):
        """List all documents in the knowledge base"""
        if not self.metadata_file.exists():
            print("❌ No knowledge base found. Process some PDFs first.")
            return
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print("\n" + "="*60)
        print("📚 DOCUMENTS IN KNOWLEDGE BASE")
        print("="*60)
        
        source_files = metadata.get('source_files', [metadata.get('source_file', '')])
        if source_files:
            for i, doc in enumerate(source_files, 1):
                print(f"{i}. {doc}")
        else:
            print("No documents found")
        
        print(f"\n📊 Statistics:")
        print(f"  Total documents: {len(source_files)}")
        print(f"  Total chunks: {metadata.get('num_chunks', 0)}")
        print(f"  Embedding model: {metadata.get('embedding_model', 'unknown')}")
        print("="*60)
    
    def backup_knowledge_base(self, backup_name=None):
        """Create a backup of the current knowledge base"""
        if not self.vector_store_path.exists():
            print("❌ No knowledge base to backup")
            return
        
        # Create backup directory
        self.backup_path.mkdir(exist_ok=True)
        
        # Generate backup name
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_dir = self.backup_path / backup_name
        
        # Copy vector store
        shutil.copytree(self.vector_store_path, backup_dir)
        print(f"✅ Backup created: {backup_dir}")
        
        # Save backup info
        info_file = backup_dir / "backup_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "name": backup_name
            }, f)
    
    def restore_backup(self, backup_name):
        """Restore a backup"""
        backup_dir = self.backup_path / backup_name
        
        if not backup_dir.exists():
            print(f"❌ Backup not found: {backup_name}")
            self.list_backups()
            return
        
        # Backup current state first
        if self.vector_store_path.exists():
            self.backup_knowledge_base("pre_restore_backup")
        
        # Remove current vector store
        if self.vector_store_path.exists():
            shutil.rmtree(self.vector_store_path)
        
        # Restore from backup
        shutil.copytree(backup_dir, self.vector_store_path)
        print(f"✅ Restored from backup: {backup_name}")
    
    def list_backups(self):
        """List all available backups"""
        if not self.backup_path.exists():
            print("❌ No backups found")
            return
        
        backups = [d for d in self.backup_path.iterdir() if d.is_dir()]
        
        if not backups:
            print("❌ No backups found")
            return
        
        print("\n" + "="*60)
        print("💾 AVAILABLE BACKUPS")
        print("="*60)
        
        for backup in sorted(backups):
            info_file = backup / "backup_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                print(f"  - {backup.name} (Created: {info.get('timestamp', 'unknown')})")
            else:
                print(f"  - {backup.name}")
        print("="*60)
    
    def get_chunk_samples(self, num_samples=3):
        """Show sample chunks from the knowledge base"""
        if not self.metadata_file.exists():
            print("❌ No knowledge base found")
            return
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        chunks = metadata.get('chunks', [])
        if not chunks:
            print("❌ No chunks found")
            return
        
        print("\n" + "="*60)
        print("📝 SAMPLE CHUNKS FROM KNOWLEDGE BASE")
        print("="*60)
        
        import random
        samples = random.sample(chunks, min(num_samples, len(chunks)))
        
        for i, chunk in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print("-" * 40)
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print("="*60)
    
    def export_metadata(self, output_file="knowledge_base_info.json"):
        """Export knowledge base metadata"""
        if not self.metadata_file.exists():
            print("❌ No knowledge base found")
            return
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create export data (without the actual chunks for size)
        export_data = {
            "source_files": metadata.get('source_files', []),
            "num_chunks": metadata.get('num_chunks', 0),
            "embedding_model": metadata.get('embedding_model', ''),
            "export_date": datetime.now().isoformat(),
            "chunk_preview": metadata.get('chunks', [])[:5]  # First 5 chunks only
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Metadata exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Manage RAG Knowledge Base Documents")
    parser.add_argument('action', choices=['list', 'backup', 'restore', 'list-backups', 
                                          'samples', 'export'],
                       help='Action to perform')
    parser.add_argument('--name', help='Backup name (for backup/restore)')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples to show')
    
    args = parser.parse_args()
    
    manager = DocumentManager()
    
    if args.action == 'list':
        manager.list_documents()
    elif args.action == 'backup':
        manager.backup_knowledge_base(args.name)
    elif args.action == 'restore':
        if not args.name:
            print("❌ Please specify backup name with --name")
            manager.list_backups()
        else:
            manager.restore_backup(args.name)
    elif args.action == 'list-backups':
        manager.list_backups()
    elif args.action == 'samples':
        manager.get_chunk_samples(args.samples)
    elif args.action == 'export':
        manager.export_metadata()

if __name__ == "__main__":
    main()