# test_file_transfer.py - Test script for file transfer system
# Place this in your project root directory and run: python test_file_transfer.py

import os
import sys
import django
from datetime import datetime

# Setup Django environment
sys.path.append('.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')
django.setup()

# Import your models and services
from ml_api.models import CustomUser, SimpleInspection, InspectionRecord
from ml_api.file_transfer_service import FileTransferService
from ml_api.nip_file_generator import NipFileGenerator
from ml_api.external_server_client import ExternalServerClient

def test_complete_system():
    """
    Complete test of the file transfer system
    """
    print("="*80)
    print("🚀 STARTING COMPLETE FILE TRANSFER SYSTEM TEST")
    print("="*80)
    
    try:
        # Test 1: Initialize services
        print(f"\n1️⃣ TESTING SERVICE INITIALIZATION")
        print("-" * 50)
        
        transfer_service = FileTransferService()
        nip_generator = NipFileGenerator()
        server_client = ExternalServerClient()
        
        print("✅ All services initialized successfully")
        
        # Test 2: Server connection test
        print(f"\n2️⃣ TESTING SERVER CONNECTION")
        print("-" * 50)
        
        connection_success, connection_msg = server_client.test_connection()
        print(f"Connection result: {'✅ SUCCESS' if connection_success else '❌ FAILED'}")
        print(f"Message: {connection_msg}")
        
        # Test 3: Server communication test
        print(f"\n3️⃣ TESTING SERVER COMMUNICATION")
        print("-" * 50)
        
        comm_success, comm_response, comm_msg = server_client.send_test_payload()
        print(f"Communication result: {'✅ SUCCESS' if comm_success else '❌ FAILED'}")
        print(f"Message: {comm_msg}")
        if comm_response:
            print(f"Server response: {comm_response}")
        
        # Test 4: Create mock inspection for testing
        print(f"\n4️⃣ CREATING MOCK INSPECTION DATA")
        print("-" * 50)
        
        # Get or create test user
        test_user, created = CustomUser.objects.get_or_create(
            username='test_transfer_user',
            defaults={
                'email': 'test@transfer.com',
                'role': 'user'
            }
        )
        
        if created:
            test_user.set_password('testpass123')
            test_user.save()
            print("✅ Created test user: test_transfer_user")
        else:
            print("✅ Using existing test user: test_transfer_user")
        
        # Create test inspection with OK status
        test_qr = f"TEST_QR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test_inspection = SimpleInspection.objects.create(
            user=test_user,
            image_id=test_qr,
            filename=f"{test_qr}_test.jpg",
            overall_result='PASS',  # This will trigger OK status
            nut1_status='PRESENT',
            nut2_status='PRESENT',
            nut3_status='PRESENT',
            nut4_status='PRESENT',
            processing_time=1.25
        )
        
        print(f"✅ Created test inspection:")
        print(f"   - QR Code: {test_qr}")
        print(f"   - Status: {test_inspection.overall_result}")
        print(f"   - All nuts: PRESENT")
        
        # Test 5: NIP file generation
        print(f"\n5️⃣ TESTING .NIP FILE GENERATION")
        print("-" * 50)
        
        nip_success, nip_path, nip_msg = nip_generator.create_nip_file(test_inspection)
        print(f"NIP generation result: {'✅ SUCCESS' if nip_success else '❌ FAILED'}")
        print(f"Message: {nip_msg}")
        
        if nip_success:
            print(f"Generated file: {nip_path}")
            
            # Read and display .nip content
            read_success, nip_content, read_msg = nip_generator.read_nip_file(nip_path)
            if read_success:
                print(f"📄 .NIP File Content Preview:")
                print(f"   - QR Code: {nip_content['metadata']['qr_code']}")
                print(f"   - Overall Status: {nip_content['inspection_results']['overall_status']}")
                print(f"   - Nuts OK: {nip_content['inspection_results']['summary']['nuts_ok']}")
                print(f"   - Nuts NG: {nip_content['inspection_results']['summary']['nuts_ng']}")
        
        # Test 6: Complete file transfer process
        print(f"\n6️⃣ TESTING COMPLETE FILE TRANSFER PROCESS")
        print("-" * 50)
        
        transfer_success, transfer_msg, transfer_details = transfer_service.process_ok_status_change(test_inspection)
        
        print(f"Transfer result: {'✅ SUCCESS' if transfer_success else '❌ FAILED'}")
        print(f"Message: {transfer_msg}")
        print(f"Details: {transfer_details}")
        
        # Test 7: File statistics
        print(f"\n7️⃣ TESTING FILE STATISTICS")
        print("-" * 50)
        
        stats = transfer_service.get_transfer_statistics()
        if 'error' not in stats:
            print("📊 Current Statistics:")
            print(f"   - Total Attempts: {stats['transfer_stats']['total_attempts']}")
            print(f"   - Successful: {stats['transfer_stats']['successful']}")
            print(f"   - Failed: {stats['transfer_stats']['failed']}")
            print(f"   - Pending: {stats['transfer_stats']['pending']}")
            print(f"   - Success Rate: {stats['transfer_stats']['success_rate']}%")
        else:
            print(f"❌ Statistics error: {stats['error']}")
        
        # Test 8: Retry mechanism (if there are failed transfers)
        print(f"\n8️⃣ TESTING RETRY MECHANISM")
        print("-" * 50)
        
        failed_files = nip_generator.get_failed_files()
        if failed_files:
            print(f"Found {len(failed_files)} failed files to test retry")
            retry_results = transfer_service.retry_failed_transfers()
            print(f"Retry results: {retry_results}")
        else:
            print("✅ No failed files found - retry mechanism available")
        
        # Test 9: System test
        print(f"\n9️⃣ RUNNING COMPREHENSIVE SYSTEM TEST")
        print("-" * 50)
        
        system_test_results = transfer_service.test_system()
        
        if 'error' not in system_test_results:
            print("🧪 System Test Results:")
            print(f"   - Server Connection: {'✅' if system_test_results['server_connection'] else '❌'}")
            print(f"   - NIP Generation: {'✅' if system_test_results['nip_generation'] else '❌'}")
            print(f"   - File Transfer: {'✅' if system_test_results['file_transfer'] else '❌'}")
            print(f"   - Overall Result: {'✅ PASS' if system_test_results['overall'] else '❌ FAIL'}")
        else:
            print(f"❌ System test error: {system_test_results['error']}")
        
        # Test 10: Configuration display
        print(f"\n🔟 CURRENT SYSTEM CONFIGURATION")
        print("-" * 50)
        
        server_info = server_client.get_server_info()
        print("🌐 Server Configuration:")
        for key, value in server_info.items():
            print(f"   - {key}: {value}")
        
        nip_stats = nip_generator.get_file_statistics()
        print(f"\n📁 File Statistics:")
        for folder, count in nip_stats.items():
            print(f"   - {folder}: {count} files")
        
        # Test Summary
        print(f"\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80)
        
        test_results = {
            'service_init': True,
            'server_connection': connection_success,
            'server_communication': comm_success,
            'nip_generation': nip_success,
            'file_transfer': transfer_success,
            'system_test': system_test_results.get('overall', False) if 'error' not in system_test_results else False
        }
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"❌ Tests Failed: {total_tests - passed_tests}/{total_tests}")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   - {test_name.replace('_', ' ').title()}: {status}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! File transfer system is working correctly.")
        else:
            print(f"\n⚠️ Some tests failed. Check the logs above for details.")
        
        # Cleanup option
        print(f"\n🧹 CLEANUP")
        print("-" * 50)
        
        cleanup_choice = input("Do you want to clean up test data? (y/n): ").lower()
        if cleanup_choice == 'y':
            # Delete test inspection
            test_inspection.delete()
            print(f"🗑️ Deleted test inspection: {test_qr}")
            
            # Clean up test user if created
            if created:
                test_user.delete()
                print(f"🗑️ Deleted test user: test_transfer_user")
            
            print("✅ Cleanup completed")
        else:
            print("ℹ️ Test data preserved")
        
        return test_results
        
    except Exception as e:
        print(f"\n💥 TEST FAILED WITH ERROR:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """
    Test individual components separately
    """
    print("\n" + "="*80)
    print("🔧 TESTING INDIVIDUAL COMPONENTS")
    print("="*80)
    
    # Test NIP Generator only
    print(f"\n🔧 Testing NIP File Generator...")
    try:
        nip_gen = NipFileGenerator()
        stats = nip_gen.get_file_statistics()
        print(f"✅ NIP Generator working - File counts: {stats}")
    except Exception as e:
        print(f"❌ NIP Generator error: {e}")
    
    # Test Server Client only
    print(f"\n🔧 Testing External Server Client...")
    try:
        server_client = ExternalServerClient()
        info = server_client.get_server_info()
        print(f"✅ Server Client working - Target: {info['full_url']}")
        
        # Test connection
        conn_success, conn_msg = server_client.test_connection()
        print(f"Connection: {'✅' if conn_success else '❌'} - {conn_msg}")
        
    except Exception as e:
        print(f"❌ Server Client error: {e}")
    
    # Test File Transfer Service only
    print(f"\n🔧 Testing File Transfer Service...")
    try:
        transfer_service = FileTransferService()
        stats = transfer_service.get_transfer_statistics()
        if 'error' not in stats:
            print(f"✅ Transfer Service working - Success rate: {stats['transfer_stats']['success_rate']}%")
        else:
            print(f"⚠️ Transfer Service warning: {stats['error']}")
    except Exception as e:
        print(f"❌ Transfer Service error: {e}")

def interactive_test():
    """
    Interactive test mode for manual testing
    """
    print("\n" + "="*80)
    print("🎮 INTERACTIVE TEST MODE")
    print("="*80)
    
    while True:
        print(f"\nChoose a test option:")
        print(f"1. Test server connection")
        print(f"2. Send test payload")
        print(f"3. Generate test .nip file")
        print(f"4. Check file statistics")
        print(f"5. Retry failed transfers")
        print(f"6. Update server settings")
        print(f"7. Run complete system test")
        print(f"8. Exit")
        
        choice = input(f"\nEnter your choice (1-8): ").strip()
        
        try:
            if choice == '1':
                client = ExternalServerClient()
                success, msg = client.test_connection()
                print(f"Result: {'✅' if success else '❌'} - {msg}")
                
            elif choice == '2':
                client = ExternalServerClient()
                success, response, msg = client.send_test_payload()
                print(f"Result: {'✅' if success else '❌'} - {msg}")
                if response:
                    print(f"Response: {response}")
                    
            elif choice == '3':
                # Create mock inspection
                mock_inspection = type('MockInspection', (), {
                    'image_id': f'INTERACTIVE_TEST_{datetime.now().strftime("%H%M%S")}',
                    'overall_result': 'PASS',
                    'nut1_status': 'PRESENT',
                    'nut2_status': 'PRESENT', 
                    'nut3_status': 'PRESENT',
                    'nut4_status': 'PRESENT',
                    'created_at': datetime.now(),
                    'processing_time': 1.0,
                    'user': type('MockUser', (), {'username': 'interactive_user'})()
                })()
                
                generator = NipFileGenerator()
                success, path, msg = generator.create_nip_file(mock_inspection)
                print(f"Result: {'✅' if success else '❌'} - {msg}")
                
            elif choice == '4':
                service = FileTransferService()
                stats = service.get_transfer_statistics()
                if 'error' not in stats:
                    print("📊 Statistics:")
                    print(f"   - Success Rate: {stats['transfer_stats']['success_rate']}%")
                    print(f"   - Pending: {stats['nip_files']['pending']}")
                    print(f"   - Sent: {stats['nip_files']['sent']}")
                    print(f"   - Failed: {stats['nip_files']['failed']}")
                else:
                    print(f"❌ Error: {stats['error']}")
                    
            elif choice == '5':
                service = FileTransferService()
                results = service.retry_failed_transfers()
                if 'error' not in results:
                    print(f"Retry completed: {results['successful']} successful, {results['failed']} failed")
                else:
                    print(f"❌ Error: {results['error']}")
                    
            elif choice == '6':
                current_ip = input("Enter new server IP (press enter to keep current): ").strip()
                current_port = input("Enter new server port (press enter to keep current): ").strip()
                
                client = ExternalServerClient()
                if current_ip or current_port:
                    client.update_server_settings(
                        server_ip=current_ip if current_ip else None,
                        server_port=int(current_port) if current_port else None
                    )
                    print("✅ Settings updated")
                else:
                    print("ℹ️ No changes made")
                    
                # Show current settings
                info = client.get_server_info()
                print(f"Current server: {info['full_url']}")
                
            elif choice == '7':
                test_complete_system()
                
            elif choice == '8':
                print("👋 Exiting interactive test mode")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"💥 Error: {str(e)}")
        
        input(f"\nPress Enter to continue...")

if __name__ == "__main__":
    print("🧪 FILE TRANSFER SYSTEM TEST SUITE")
    print("="*80)
    
    test_mode = input("Choose test mode:\n1. Complete automated test\n2. Individual component tests\n3. Interactive test mode\n\nEnter choice (1-3): ").strip()
    
    if test_mode == '1':
        test_complete_system()
    elif test_mode == '2':
        test_individual_components()
    elif test_mode == '3':
        interactive_test()
    else:
        print("❌ Invalid choice. Running complete test by default.")
        test_complete_system()
    
    print(f"\n🏁 Test session completed!")
    print(f"📝 Check the terminal output above for detailed results.")
    print(f"📁 Check your media/nip_files/ folder for generated files.")
    print(f"📋 Check your media/transfer_logs/ folder for transfer logs.")