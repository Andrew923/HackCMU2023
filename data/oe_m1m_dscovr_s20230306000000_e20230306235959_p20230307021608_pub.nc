CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230306000000_e20230306235959_p20230307021608_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-07T02:16:08.086Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-06T00:00:00.000Z   time_coverage_end         2023-03-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxkC|�  �          @��R@��\��{�����.ffC�T{@��\>�������0  @?\)                                    BxkC�&  T          @�
=@�녿B�\����(Q�C�J=@�녽�\)�����MG�C���                                    BxkC��  �          @�\)@�(���
=�����7�C���@�(���p����\�EG�C��q                                    BxkC�r  "          @��R@��\��\�#�
��  C���@��\��33���H�;33C��f                                    BxkC�  T          @��R@�{��p��\)����C���@�{��33���
�33C��                                    BxkCž  �          @�p�@����
=������C�  @����33�Y�����C�O\                                    BxkC�d  �          @��
@��R��ff=�?�(�C��q@��R���z�����C�:�                                    BxkC�
  �          @�z�@�  �*=q>�@�C�H@�  � �׿�G��#
=C��3                                    BxkC�  �          @��R@��H�mp�?.{@�G�C��f@��H�`�׿����V�RC�K�                                    BxkD V  "          @�p�@��R�`  ?=p�@�RC��\@��R�W����H�=�C�7
                                    BxkD�  �          @�(�@�{�J=q>�\)@-p�C���@�{�5����`z�C�)                                    BxkD�  T          @�(�@�\)�]p�>��
@G�C��f@�\)�G
=��ff�t��C�T{                                    BxkD,H  
�          @�(�@����p��=�G�?��
C�&f@����P  �������C�'�                                    BxkD:�  �          @��@�
=�J�H�����
C���@�
=�+��ٙ���C��3                                    BxkDI�  �          @��@�ff�Fff���
�L��C���@�ff�(Q��33��
=C��                                    BxkDX:  T          @�(�@�ff�J�H�#�
���
C���@�ff�,(�����ffC��{                                    BxkDf�  �          @��\@��\�  �.{�ٙ�C���@��\���ÿ�=q�T(�C��\                                    BxkDu�  �          @�@�zῢ�\�����8��C�aH@�z���H�У���{C�n                                    BxkD�,  
�          @�(�@�zῺ�H�����C�p�@�z῀  ��z��>�HC���                                    BxkD��  
�          @�=q@�G���    <#�
C�@ @�G����^�R��C�|)                                    BxkD�x  
Z          @���@��H�
=q=u?+�C���@��H��\)��=q�<  C�o\                                    BxkD�  �          @�Q�@�ff��Q�˅��33C��@�ff�O\)���ӅC��                                    BxkD��  �          @��@�G���33��G���\)C�S3@�G����Ϳu�\)C���                                    BxkD�j  
�          @�G�@�����H<�>�{C��R@����Q쿀  �&�HC���                                    BxkD�  �          @���@��R��H�#�
�ǮC��)@��R��\��ff�Z{C��                                    BxkD�  T          @���@����
�H?
=@ÅC�XR@����
=q�#�
���
C�h�                                    BxkD�\  �          @��@�z��#�
>��?���C�/\@�z���׿��H�J=qC���                                    BxkE  �          @�33@����7�>��@(��C��=@����%����
�T  C��R                                    BxkE�  �          @��@�
=�@  �L����C��=@�
=��Ϳ޸R��
=C�J=                                    BxkE%N  �          @�(�@����8Q�>��@���C�u�@����*�H��z��?�C�j=                                    BxkE3�  �          @��@����<��>�p�@s�
C���@����-p���p��MC��                                    BxkEB�  �          @��H@��H�(��?5@��
C���@��H�'��E����RC���                                    BxkEQ@  �          @��H@�
=�   >��
@Tz�C��@�
=�33���
�)�C���                                    BxkE_�  �          @�=q@�ff��z�aG��33C���@�ff��G���Q��E��C��                                    BxkEn�  �          @���@��H�J�H?Q�A
�\C��\@��H�HQ�u�!C��                                    BxkE}2  �          @�=q@�Q��@  ?�AC\)C�/\@�Q��J�H�\)���HC�q�                                    BxkE��  �          @��@z=q�X��?xQ�A)C��f@z=q�Y���p���$z�C�}q                                    BxkE�~  �          @�@��H�   ��Q��eC���@��H������{��p�C���                                    BxkE�$  �          @�z�@�ff>�������F{@0  @�ff?Y���c�
�{AG�                                    BxkE��  �          @��@�����R����/\)C�
=@��׿����z���p�C�|)                                    BxkE�p  �          @�{@��R��
=>�33@b�\C���@��R��33��ff����C���                                    BxkE�  �          @�ff@���˅?!G�@ʏ\C���@����Q쾔z��=p�C�Y�                                    BxkE�  �          @�
=@��\�   >\@u�C���@��\��z�8Q���RC�#�                                    BxkE�b  �          @�
=@�
=��ff?�@��\C�R@�
=���;�33�a�C���                                    BxkF  �          @�Q�@����{>Ǯ@|(�C��@������!G��ə�C��{                                    BxkF�  �          @�Q�@�p���33>�  @\)C�]q@�p���  �E����RC�                                    BxkFT  �          @�\)@�
=��33��\)�1�C��)@�
=���\����0Q�C�|)                                    BxkF,�  �          @��R@�����>#�
?�33C�G�@�����\)�p����C�O\                                    BxkF;�  �          @��@��
��Q�>�@�(�C�
@��
��33�(���33C�H�                                    BxkFJF  �          @���@�(���
>�\)@/\)C�� @�(���z�Tz����C�:�                                    BxkFX�  �          @��@�{�#�
<�>�{C��@�{�p���ff�N�HC�}q                                    BxkFg�  T          @�
=@���\)>.{?ٙ�C�o\@���   ��G��$  C��=                                    BxkFv8  �          @���@�{��ff?�R@�p�C�ٚ@�{��{�����~�RC��\                                    BxkF��  �          @���@���
=��p��i��C�=q@����H����Q�C�|)                                    BxkF��  �          @��@�\)�\)��Q�Y��C�>�@�\)���{�Y��C��                                    BxkF�*  �          @���@�{��������C�� @�{��(�����'
=C�q�                                    BxkF��  "          @��
@�(��5����
�uC��\@�(��=q��p��jffC�j=                                    BxkF�v  �          @�(�@��
�7������Q�C�\)@��
�Q��{�~=qC��f                                    BxkF�  S          @��
@����0��=�\)?333C��@����=q��{�X  C�z�                                    BxkF��  "          @�@�ff�3�
=�Q�?^�RC��=@�ff�p���\)�W
=C�Y�                                    BxkF�h  
�          @���@�  �*=q�W
=��C���@�  �
�H�Ǯ�u�C��H                                    BxkF�  	�          @��@��H�O\)?�\@�  C��@��H�B�\��p��B�\C��\                                    BxkG�  
(          @���@�
=�R�\?:�H@�\)C���@�
=�L(������)p�C��3                                    BxkGZ  
�          @�
=@���\��?B�\@�C�7
@���Vff�����5�C��H                                    BxkG&   "          @���@����\��?uA��C�Y�@����\�Ϳs33�  C�XR                                    BxkG4�  T          @��@��q�?z�@�ffC���@��b�\��Q��dQ�C��                                    BxkGCL  �          @��
@���y��������C��@���U�����z�C�T{                                    BxkGQ�  �          @��
@�  �}p�?#�
@���C���@�  �n�R��p��k33C�`                                     BxkG`�  T          @�33@�33�qG�?aG�A
�RC���@�33�l(���
=�;33C��R                                    BxkGo>  �          @�33@�ff�j�H?^�RA	p�C�=q@�ff�fff�����333C��                                    BxkG}�  
�          @�@�������>��H@�
=C�E@����n�R��z���Q�C�t{                                    BxkG��  �          @��
@���(��?(�@�
=C��@���%��Tz���C�+�                                    BxkG�0  �          @Å@����z�H?(��@�\)C��H@����mp���Q��\��C�n                                    BxkG��  �          @�\)@�G��xQ�?p��A\)C���@�G��tz῕�-�C���                                    BxkG�|  �          @�@��
�h��?��A�HC��q@��
�j=q�xQ��z�C���                                    BxkG�"  �          @�ff@�ff�c�
?�ffA��C�q@�ff�fff�h����
C��R                                    BxkG��  T          @Å@�ff�a�?\)@�G�C�B�@�ff�TzῨ���G\)C��                                    BxkG�n  �          @���@����C�
?!G�@�Q�C���@����=p����\��C�{                                    BxkG�  �          @�Q�@�(��;�?:�H@��C�|)@�(��9���^�R�
=C��                                    BxkH�  "          @��@��H�]p�?��@��RC�� @��H�Q녿�  �G33C�8R                                    BxkH`  �          @�\)@����e�����{C��{@����C�
�������\C��=                                    BxkH  �          @�{@�33�U����#33C��3@�33�0  ��
=����C�<)                                    BxkH-�  T          @���@�ff�5�5��=qC��@�ff��33��p�C���                                    BxkH<R  �          @�G�@�Q��J�H��Q�p��C�p�@�Q��+���Q����RC��)                                    BxkHJ�  �          @��@����   �O\)���C�&f@��ÿ����(���=qC���                                    BxkHY�  �          @�  @��Ϳ��^�R�p�C��@��Ϳ�  �˅��33C���                                    BxkHhD  �          @�Q�@�
=����p��w�C��@�
=��G����Dz�C�<)                                    BxkHv�  "          @�33@��\��\)��\)�0��C��\@��\��{�aG����C��
                                    BxkH��  "          @��\@�  ��@G�A�z�C���@�  �/\)?��
A#\)C���                                    BxkH�6  "          @���@����33@	��A�p�C���@���.�R?c�
A�RC���                                    BxkH��  �          @��H@�����?��HAt(�C�� @����5�<�>���C���                                    BxkH��  T          @�p�@��Ϳ�Q�?�(�A���C��@����"�\?�\@�{C��=                                    BxkH�(  �          @�\)@�Q��<�;����Q�C�� @�Q������  ��p�C�K�                                    BxkH��  �          @��@��H�   ���R�\C�f@��H�����G���C��                                    BxkH�t  �          @���@�
=�	���������C���@�
=�E��;��	�C��                                    BxkH�  �          @�@^{�333@�A���C���@^{�fff?(�@���C���                                    BxkH��  
�          @��
@j=q�J�H@
=qA��\C���@j=q�u�>�=q@<(�C���                                    BxkI	f  
�          @�
=@X���,��@S33B{C�� @X���\)?��RA~�\C�.                                    BxkI  �          @���@.�R�,(�@r�\B2�RC��\@.�R��\)?�A�G�C�e                                    BxkI&�  �          @�@P  �O\)@���BLG�C���@P  �2�\@K�B\)C���                                    BxkI5X  �          @�  @8���:=q@j=qB'  C�l�@8����33?��HA��RC��                                    BxkIC�  �          @��\@O\)�]p�@9��A�\)C���@O\)��ff?E�@�\)C��                                    BxkIR�  �          @�Q�@p  �xQ�?�
=A�33C�H@p  ���\��\)�1�C�w
                                    BxkIaJ  T          @�G�@j=q�w�@
=qA���C��\@j=q����\)�.{C�                                    BxkIo�  �          @�p�@QG��l(�@,��A��
C��@QG�����>��H@���C���                                    BxkI~�  T          @��@S�
�c33@)��A�\)C�� @S�
����?�@�C�y�                                    BxkI�<  �          @�=q@(Q��hQ�@HQ�B�C��R@(Q����R?fffA\)C���                                    BxkI��  �          @�33@w
=�l��?��A���C��@w
=���׾�
=���\C���                                    BxkI��  �          @�@h���u?�
=A�ffC���@h����G��u���C�*=                                    BxkI�.  �          @��@s�
�p  ?���A�33C��
@s�
��p�����(��C�.                                    BxkI��  T          @��
@�  �Y��?�z�A�  C���@�  �y������Q�C��H                                    BxkI�z  �          @�(�@~{�l(�?ǮA���C��@~{�~�R����(�C�z�                                    BxkI�   T          @�G�@z�H���?xQ�A�\C���@z�H��G������A�C�
=                                    BxkI��  �          @���@x�����?Y��AQ�C��\@x�����ÿ����T��C�                                      BxkJl  �          @���@�����z�=�G�?�{C�3@����i�����H���RC��H                                    BxkJ  �          @�{@|(����H�����p�C��3@|(��^�R�
=q���\C�C�                                    BxkJ�  �          @���@tz��|�;�ff��ffC��@tz��L���ff�ɮC��                                    BxkJ.^  �          @�ff@���C33?
=q@�
=C���@���:�H����2�RC�AH                                    BxkJ=  �          @��@|(���\@C�
B
33C�Y�@|(��Q�?�33A��\C��                                    BxkJK�  �          @�{@�=q�?ٙ�A���C�k�@�=q�8Q�>��
@Z=qC���                                    BxkJZP  T          @��
@�Q���?
=@�G�C���@�Q��ff�(�����HC��)                                    BxkJh�  "          @��@�p��%>���@UC��\@�p�������\�733C���                                    BxkJw�  T          @��
@�
=��\)>��
@c�
C���@�
=���׾�\)�FffC��R                                    BxkJ�B  �          @��H@�녾aG���\)�@  C���@�녾#�
�.{���C�R                                    BxkJ��  T          @�=q@���>u�����z�@1G�@���>�\)�#�
��Q�@N�R                                    BxkJ��  �          @�33@��þ����H����C�Ff@��þ8Q�(����Q�C���                                    BxkJ�4  �          @�G�@�  >��Ϳ\����@���@�  ?��Ϳ����P��AO�                                    BxkJ��  �          @���@���?
=q���H�a@�(�@���?���J=q��HAG�
                                    BxkJπ  �          @�Q�@���?�Ϳ����v�R@Ӆ@���?��׿aG��#\)AUp�                                    BxkJ�&  �          @��@���>����R��{@��@���?�33�����C�AXz�                                    BxkJ��  
�          @���@�
=>#�
�L����R?���@�
=?���R��33@�33                                    BxkJ�r  T          @��@�p�>�
=�(���=q@�z�@�p�?+������q�@�
=                                    BxkK
  �          @���@���>�  ��33��  @3�
@���>Ǯ�B�\��@��                                    BxkK�  "          @��@�
=?W
=��{�vffA�@�
=?fff=�G�?�ffA$Q�                                    BxkK'd  �          @�ff@�33?Y�������(�A�@�33?Q�>�z�@UA��                                    BxkK6
  �          @��@�(�?z�?8Q�A��@ڏ\@�(�>#�
?h��A)?�{                                    BxkKD�  �          @���@�Q�=�Q�>��@���?}p�@�Q���>�@�ffC�q                                    BxkKSV  �          @��\@���?�=q?��\Ah  AK33@���>��
?У�A�ff@xQ�                                    BxkKa�  "          @��@��>��?�  Af�H@�G�@����  ?��AnffC��H                                    BxkKp�  T          @�p�@�  ?�?���AJ=q@���@�  ��Q�?��RA_33C���                                    BxkKH  �          @��@���z�?��\Ag�C�N@��^�R?xQ�A/�
C���                                    BxkK��  T          @��@��þ�{@��A��C�� @��ÿ�=q?��HA�G�C��                                    BxkK��  T          @�@~{?�@Mp�Bp�A�@~{���@E�BQ�C��\                                    BxkK�:  
�          @�@<�Ϳ���@XQ�B.�C�H�@<���U@ ��A�z�C���                                    BxkK��  �          @�
=@z��=q@��\Bl
=C�P�@z��^�R@A�B\)C�`                                     BxkKȆ  �          @���?�
=��z�@�  Br{C��\?�
=�u@AG�B��C�Ф                                    BxkK�,  "          @��Ϳ�G��Y��@e�B2z�Cy�3��G���ff?��RA�z�C�                                    BxkK��  �          @��R?c�
�z=q@���B4�C�ff?c�
���
?��A�  C��{                                    BxkK�x  �          @��u�Z�H@��
BP{C��H�u��33@(�A�  C���                                    BxkL  �          @�=q������@�33Bvp�CrǮ������(�@HQ�B
z�C~�                                    BxkL�  �          @�G��W
=�L��@��B��3CAs3�W
=�8��@��Bi  C}�)                                    BxkL j  �          @��H��R��R@��RB���Ca���R�J=q@��BYffC�y�                                    BxkL/  �          @�33>k��@  @��RB���C���>k��333@c33BN�RC�]q                                    BxkL=�  T          @���@�R�   @w�BD=qC�� @�R����@��A�G�C�q�                                    BxkLL\  �          @�\)�����=q@��
B���Cp�q����:�H@FffB8�RC�'�                                    BxkL[  T          @���@
=@�ff?L��A(�B�Q�@
=@a�@5�B
G�Bl�                                    BxkLi�  �          @�\)?�Q�@�G��}p��5G�B��?�Q�@�?�A���B��                                     BxkLxN  "          @��@{@ff=#�
?!G�B!ff@{?�=q?��
A�z�B                                      BxkL��  �          @�z�@h�ÿ�\)@��A�ffC�@h���   ?�(�Av=qC��                                    BxkL��  T          @��@b�\��@Aϙ�C��=@b�\�HQ�?.{A�C�Ff                                    BxkL�@  �          @��
@E�Q�@��A�C�7
@E�N�R?xQ�AD  C��                                    BxkL��  "          @�G�@{�@\)B��C�Q�@{�O\)?�=qAmp�C��3                                    BxkL��  �          @��\@K����H������C��
@K�?z����
=A%                                    BxkL�2  �          @��\?�G�?޸R�s33�r{BX{?�G�@U��!G��=qB��                                     BxkL��  �          @�G�?}p�@p��w
=�gp�B���?}p�@q��ff����B��                                     BxkL�~  �          @w�>�\)?��R�Q��j33B��>�\)@S33������  B�ff                                    BxkL�$  T          @n{�L��?���a��{B����L��@&ff�'
=�4�B�G�                                    BxkM
�  �          @�\)�ٙ�?0���\(��w�
C�ٙ�@\)�-p��/�B�k�                                    BxkMp  T          @��\��\)���^{�[33Cd�H��\)<#�
�{�33C3}q                                    BxkM(  �          @I���h���   ��(��'��Cuh��h�ÿW
=�+�ffC^h�                                    BxkM6�  �          @G���R�:�H�ff  Ceh���R>�33�(�
=C��                                    BxkMEb  
�          @J�H��Q�<��
�G�¦ǮC0s3��Q�?�p��/\)�s�\B�Q�                                    BxkMT  "          @=p���\)=��:�H§��C0���\)?�p��!G��l�\Bɔ{                                    BxkMb�  "          @Mp���\)?.{�E�{B�׾�\)@33�=q�E=qBÔ{                                    BxkMqT  �          @=p����H?���=q�oz�B�\���H@�\�\��B�8R                                    BxkM�  �          @g
=�Q�?�\)�33��C�R�Q�@�\������\C{                                    BxkM��  �          @p  ��33@��)���6�B�� ��33@J�H��=q����B�Ǯ                                    BxkM�F  "          @p�׿�@:=q��ff��
=B�ff��@K�>W
=@S�
B��                                    BxkM��  #          @hQ쿝p�@O\)�333�733Bݞ���p�@L(�?c�
Ah��B�(�                                    BxkM��  �          @Z�H�p��@G
=�.{�<Q�Bսq�p��@Dz�?Y��Al��B�.                                    BxkM�8  T          @'���?�ff������BԔ{��@G���G��,��B�#�                                    BxkM��  "          @�p�?�G�?p���Q��j��A�=q?�G�@
=�{� {BUz�                                    BxkM�  T          @QG�?�ff?n{�+��_�
A�z�?�ff@�
��Q���
BT
=                                    BxkM�*  
�          @�Q�@-p�>�=q�s33�Y�@���@-p�@   �O\)�/�\B�H                                    BxkN�  �          @��@Q��G��*�H�)G�C�Q�@Q����S33�a�C��                                    BxkNv  �          @�\)@	���<����
��C�  @	�����L(��Fz�C��                                    BxkN!  
�          @�  @5�7
=������
C�o\@5���H�:=q�%  C���                                    BxkN/�  "          @�z�@h�ÿ�ff�  ��{C���@h�ÿ\)�5���C���                                    BxkN>h  �          @�{@��\>��"�\����?ٙ�@��\?��
�(���G�A�                                      BxkNM  
Z          @�Q�@XQ�?��\�U��*�HA�
=@XQ�@*=q����33B��                                    BxkN[�  "          @�33@,(����
�<(��.(�C�/\@,(��#�
�S�
�K��C���                                    BxkNjZ  
�          @��@U�>�{�g
=�<ff@��@U�?��H�C33�  A�p�                                    BxkNy   �          @�p�@!��z��|(��IQ�C�XR@!논���ff�q��C���                                    BxkN��  
[          @��
@\)?:�H��\)�i�HA�\)@\)@%�[��.��B8p�                                    BxkN�L  
�          @��@/\)?�����  �VA��\@/\)@1��Dz��B5�                                    BxkN��  �          @��H@�
?��������U�B ��@�
@e��-p����Bd��                                    BxkN��  �          @���?�p�@9����Q��Gz�Bl��?�p�@���\)����B�8R                                    BxkN�>  
�          @���@�H@$z��^{�2{B:�@�H@w�������Bg�                                    BxkN��  �          @��@+�?���"�\��
A�\@+�@!G�������ffB-(�                                    BxkNߊ  
�          @mp�?W
=@%�8Q��n�RB�?W
=@��?��\A�G�B�33                                    BxkN�0  S          @�p�@���8Q�?�Q�A�G�C��@���N{������C���                                    BxkN��  #          @��H@333��ff@��B��C���@333�%�?��A�=qC��{                                    BxkO|  �          @��
@'
=��Q�@L��B7p�C��=@'
=�<��@A߮C��
                                    BxkO"  �          @��
@(�����@QG�BD�C��f@(��1G�@��A��C���                                    BxkO(�  U          @J=q?�׿c�
@
=BBffC�ff?�׿���?��HB\)C��{                                    BxkO7n  T          @���@\(��!�@�G�B-�C�Ф@\(����H@{A�ffC�                                    BxkOF  �          @���@q��.{@s�
B�C�R@q�����@
�HA��C�(�                                    BxkOT�  T          @�ff@����-p�@Y��B�C�q@����|��?���A���C��q                                    BxkOc`  
�          @���@��R�   @S33B�RC�� @��R�n{?���A��C�Ff                                    BxkOr  
Z          @�z�@�p��$z�@Q�B{C�1�@�p��p��?��
A�ffC��R                                    BxkO��  
�          @�(�@����R@C�
A��C�33@���e?У�A���C�G�                                    BxkO�R  "          @�33@���	��@"�\A��
C�˅@���C�
?��AP��C���                                    BxkO��  
�          @�=q@�����?�33A�ffC�S3@����
?xQ�A��C��                                    BxkO��  �          @��@��\�{@=qA���C��=@��\�C�
?�A8Q�C���                                    BxkO�D  �          @��\@����?�z�A��C�*=@���*�H?O\)@��C�4{                                    BxkO��  "          @��\@�
=�@(�A�G�C��R@�
=�6ff?��A$(�C�
                                    BxkOؐ  �          @���@��\�   @p�A��C�C�@��\�1G�?��A*=qC��{                                    BxkO�6  "          @�p�@�����H@G�B{C�P�@����c33?�(�A�
=C�9�                                    BxkO��  "          @�{@���Q�@0��A�(�C�~�@���Vff?�A]�C�%                                    BxkP�  �          @�ff@����[�@(��A�=qC��H@�����  ?Y��A(�C��
                                    BxkP(  �          @�p�@�ff�e�?���A��C���@�ff��=q>.{?�z�C���                                    BxkP!�  �          @�{@����O\)@�RA�Q�C�o\@�����  ?L��@�\)C�u�                                    BxkP0t  
(          @���@�Q��[�@
�HA�G�C��R@�Q���G�>�G�@���C�H�                                    BxkP?  �          @��@��5�@z�A��C�ff@��\��?(�@��C�Ǯ                                    BxkPM�  
�          @�(�@�G��AG�@
�HA�C�(�@�G��j�H?�R@��HC���                                    BxkP\f  �          @�z�@�
=�<��@�HA���C�N@�
=�mp�?aG�A
ffC�'�                                    BxkPk  T          @�(�@����<(�@33A���C�u�@����i��?G�@���C���                                    BxkPy�  T          @�
=@�
=�B�\@   A�(�C��@�
=�u�?h��A�C��)                                    BxkP�X  "          @�33@�(��L(�?���Az{C���@�(��c�
<��
>B�\C�5�                                    BxkP��  �          @�(�@���<(�?���A�Q�C�
@���\(�>�33@[�C��                                    BxkP��  T          @��
@�=q�;�@{A�=qC���@�=q�fff?8Q�@��HC���                                    BxkP�J  �          @�ff@���C33@5A��
C�)@���~�R?�(�A>{C�c�                                    BxkP��  
Z          @��@���W
=?��\A$��C���@���]p��
=q��C�,�                                    BxkPі  S          @�33@��
�aG���p��fffC�^�@��
�@�׿�{���RC�w
                                    BxkP�<  U          @��
@�{�L��?���Aep�C��
@�{�`�׽��
�E�C��3                                    BxkP��  T          @�(�@�{�1�@
�HA���C��f@�{�\��?@  @�C��{                                    BxkP��  
�          @�p�@�33�<(�@.{A�
=C��)@�33�u�?�
=A8��C�W
                                    BxkQ.  �          @�\)@�p��^{@�HA���C�"�@�p���p�?0��@�C�}q                                    BxkQ�  T          @�Q�@����Q�?��HA�z�C�
=@����s�
>�Q�@\(�C��)                                    BxkQ)z  �          @��R@����o\)?޸RA��C�l�@�����(��#�
��
=C�
=                                    BxkQ8   
(          @�=q@�\)��z�@\��B�C�� @�\)�L��@33A�G�C�o\                                    BxkQF�  
�          @�\)@��� ��@O\)B33C��@���i��?���A�=qC��                                    BxkQUl  �          @�Q�@�z�� ��@c�
B33C�c�@�z��r�\@��A���C���                                    BxkQd  �          @�
=@����@g�B�
C�"�@��b�\@�A��\C��                                    BxkQr�  
�          @���@��H�*�H@5�A�{C��@��H�g�?�z�AYp�C��)                                    BxkQ�^  �          @���@������@1�A�ffC�n@����U?�  AvffC�H                                    BxkQ�  �          @�ff�aG���@��B��)Cw(��aG��1�@�p�Bl�C��{                                    BxkQ��  �          @��?\)��  @�
=B�z�C���?\)�e@a�B0�C�j=                                    BxkQ�P  "          @�@Dz�5@�  BN��C���@Dz��@Tz�B"�
C�aH                                    BxkQ��  �          @�=q@���  @�
=Bg��C��R@��P  @Z=qB!G�C�
=                                    BxkQʜ  
�          @�  ?s33�u�@tz�B0�\C���?s33���\?���A�33C�K�                                    BxkQ�B  "          @��H>���[�@��RBP��C��>�����@%Aڏ\C��3                                    BxkQ��  T          @�33?   ��@�  B�k�C���?   ��p�@o\)B �C�7
                                    BxkQ��  �          @�{?@  �ٙ�@��
B�ffC�ٚ?@  �z=q@�(�B>�C�g�                                    BxkR4  !          @�{?����@��\B��\C��{?���w
=@��B=Q�C��f                                    BxkR�  #          @��?k���G�@���B���C�?k��{�@���B;=qC���                                    BxkR"�  �          @��H?�  ��Q�@�=qB��C�e?�  �\)@���B/Q�C���                                    BxkR1&  
�          @�z�?�
=��ff@��
B�u�C�=q?�
=�j�H@j�HB+C���                                    BxkR?�  T          @�(�?�=q�+�@�  Be{C��q?�=q��=q@>{B�RC�
=                                    BxkRNr  �          @�\)?   �Q�@��Bx��C��?   ���\@L(�B33C�~�                                    BxkR]  
�          @���=��
��@�B�  C�C�=��
�g�@_\)B/�\C��                                    BxkRk�  
�          @�녾�����
@��HB�B�C�<)����Dz�@h��BF��C��3                                    BxkRzd  T          @���@���z�@�G�Bj��C��=@��5�@Z�HB+�HC��
                                    BxkR�
  
�          @�{@��R���@.�RA��C���@��R�HQ�?���A�ffC�q�                                    BxkR��  �          @�{@�z��(�@=qAɮC�aH@�z��?\)?��AR�HC��R                                    BxkR�V  �          @�(�@���(�?��AX��C�.@���!�>�z�@?\)C���                                    BxkR��  T          @�z�@�  �\)?��AS�C��3@�  �$z�>�  @"�\C�g�                                    BxkRâ  "          @�  @��\��=q@=qA�=qC��@��\�*=q?�Q�As33C��                                    BxkR�H  �          @��
@�=q��p�@;�A�33C��@�=q�#�
@�
A�  C�b�                                    BxkR��  �          @��@Y���W
=@w�B?��C�R@Y���ff@L(�B��C��f                                    BxkR�  
�          @���@Mp���33@{�BD{C�q@Mp��*=q@G
=BC�(�                                    BxkR�:  �          @�=q@;��
�H@aG�B/��C��)@;��Z�H@z�A�(�C�J=                                    BxkS�  
�          @�{�=p�?�33@�B�
=B�uÿ=p��W
=@���B�33Cd�=                                    BxkS�  
�          @��R��ff@G�@�Q�B�B�#׾�ff����@�p�B���C@��                                    BxkS*,  T          @�p�>aG�?�(�@��B�\B��{>aG��k�@���B��
C��)                                    BxkS8�  �          @�G�?:�H>��
@��RB�z�A���?:�H����@�z�B�8RC���                                    BxkSGx  �          @�\)?����@���B�B�C�?���\@�Q�Bq=qC��                                    BxkSV  "          @���?�녿n{@�ffB���C�R?���7
=@���BR{C��R                                    BxkSd�  �          @�Q�@"�\����@\)BM
=C�G�@"�\�X��@7
=B  C�h�                                    BxkSsj  
�          @�@���@��B��C���@�@  @~�RB=�C�e                                    BxkS�  T          @��?�׿
=q@�
=B�{C��?���   @��\BXC�z�                                    BxkS��  "          @�{@�� ��@�B^
=C�P�@��g�@O\)Bz�C��                                    BxkS�\  �          @�@�R��(�@�Q�Bb�C�K�@�R�g�@UB�RC��{                                    BxkS�  �          @���@�
�޸R@��Bn��C��@�
�\(�@aG�B%
=C�w
                                    BxkS��  �          @�ff?�z�k�@���B��HC�xR?�z��9��@�33BU�RC��                                    BxkS�N  
�          @�\)@�Ϳk�@��B�33C��
@���3�
@�(�BD��C���                                    BxkS��  
�          @�p�@k���?��A�=qC��@k���p�?�G�A�  C��q                                    BxkS�  "          @��@��
�   �   ����C�� @��
?
=q�\)���
@ᙚ                                    BxkS�@  
�          @���@����ff���
=C��@�����$z���\)C�˅                                    BxkT�  
�          @�@��
���Ϳ��
��z�C���@��
�����z���ffC�
=                                    BxkT�  �          @��@�þ\����x  C��@�ý�G��333��33C���                                    BxkT#2  
(          @�  ?n{�u��(� G�C�"�?n{@z����R�RB��{                                    BxkT1�  �          @��>�׿p����{(�C�/\>��?�����
\)B�33                                    BxkT@~  �          @�G�?����33����W
C��H?��>�  ��z�\ARff                                    BxkTO$  "          @��?�G���33���
��C��3?�G�?   ��33�=A��                                    BxkT]�  �          @��\?�
=�����p���C��?�
=?B�\����
=A���                                    BxkTlp  T          @�p�@
=�aG���\)�x  C�Ǯ@
=?�ff��ff�t��A�=q                                    BxkT{  
�          @�ff@vff��G��G����C�� @vff�Ǯ�c�
�*ffC�)                                    BxkT��  �          @���@�{�G��Ǯ����C�\@�{��p�������C�=q                                    BxkT�b  �          @�p�@���W
=>��H@�(�C�Z�@���u=�?���C��                                    BxkT�  "          @��
@�p�?   @�\A�
=@�=q@�p�����@z�A�(�C�\                                    BxkT��  
�          @�G�@�=q?���@=qA�33AR�H@�=q=L��@(��A�?��                                    BxkT�T  �          @���@��H?�p�>�
=@��HAZ{@��H?xQ�?^�RA  A-G�                                    BxkT��  �          @�Q�@�=q?u>���@�(�A,  @�=q?:�H?=p�A33A�                                    BxkT�  T          @��H@������ý��Ϳ�ffC�9�@�����=q�aG��ffC���                                    BxkT�F  �          @��\@�{>��?��A=��@���@�{<#�
?�Q�AN�\=�Q�                                    BxkT��  �          @�33@�z�>L��?�Av=q@\)@�zᾸQ�?���Apz�C��q                                    BxkU�  �          @��
@�{��?�Q�A��HC��@�{���R?˅A���C��
                                    BxkU8  �          @�  @�(�����?�33A�p�C�  @�(���\)?�{A:{C�>�                                    BxkU*�  �          @�p�@�z����?�Q�A��C��@�z��Q�?s33A!C�<)                                    BxkU9�  �          @�=q@�ff��33@AɅC�U�@�ff��H?��
A�33C�`                                     BxkUH*  �          @�p�@��
�k�@0��A�
=C���@��
��z�@��A�z�C�O\                                    BxkUV�  �          @�{@��\�&ff@:=qA���C�1�@��\���H@(�A�{C�AH                                    BxkUev  "          @�ff@��\��@U�B\)C�!H@��\�ٙ�@9��A�p�C���                                    BxkUt  �          @���@L��=#�
@��B\�?=p�@L�Ϳ�(�@���BF�C��                                    BxkU��  
�          @���@5�?(��@�Bh�AP��@5���@��\B`�HC���                                    BxkU�h  T          @��@5?&ff@���Be33AN{@5��\)@�ffB^
=C�9�                                    BxkU�  �          @���@z=q��p�@'�B�RC�K�@z=q��=q@�A�G�C��{                                    BxkU��  "          @���@����Q�u��RC��{@�����������HC���                                    BxkU�Z  "          @�\)@�G���33��Q쿈��C��H@�G����R�5� ��C�Z�                                    BxkU�   �          @��R@��Ǯ?�{Av�RC��@�����?+�@��C���                                    BxkUڦ  "          @�z�@�G��У׿G��\)C���@�G���(���=q�q�C��q                                    BxkU�L  "          @�(�@������
�^�R�p�C�,�@�����������{33C��q                                    BxkU��  "          @�p�@�(����H��������C��f@�(������p���\)C���                                    BxkV�  "          @��@��׿���?fffA"ffC��
@��׿���>u@(��C��=                                    BxkV>  �          @��@�{�\?�AT��C�
=@�{��?�@��HC�Ff                                    BxkV#�  "          @��\@�G���
=@��Aי�C�U�@�G���
=?�z�A�p�C��q                                    BxkV2�  �          @���@�녿�\)���Ϳ�Q�C��@�녿�p��(���ffC��{                                    BxkVA0  "          @��@�G���Q�?���Au��C�f@�G���{?Q�A�RC��R                                    BxkVO�  �          @��@�녿�?��A�(�C��{@�녿�Q�?�ffA�C��{                                    BxkV^|  �          @�(�@�  ?
=@  A��@�  @�  ��\)@�
A��C�1�                                    BxkVm"  
Z          @�
=@�p�=u@^�RB�?W
=@�p���p�@P  B{C��                                    BxkV{�  �          @�G�@�p�����@`��B�
C��)@�p�����@H��BQ�C��H                                    BxkV�n  �          @��@X�ÿY��@��\BLG�C��{@X�����@mp�B'Q�C�q�                                    BxkV�  
�          @���@Dzᾊ=q@��RBh��C�}q@Dz��z�@�Q�BJ�C���                                    BxkV��  
�          @���@K��У�@��HBPQ�C�y�@K��L��@j�HB��C�q�                                    BxkV�`  T          @�  @u���@fffB�C���@u�Z�H@!�A�\)C�%                                    BxkV�  T          @��@vff��{@g�B!(�C��@vff�C33@,(�A�C��f                                    BxkVӬ  �          @��\@,(���33@���B](�C�9�@,(��L(�@hQ�B$33C�f                                    BxkV�R  
�          @��@��9��@�(�B@  C�^�@����R@2�\A��C�z�                                    BxkV��  �          @��@^�R�N�R@Q�A��C���@^�R�w�?��A5p�C���                                    BxkV��  T          @�
=@����&ff?���A��C�b�@����H��?^�RAp�C��                                    BxkWD  �          @�z�@a��r�\?��@���C�}q@a��n�R�\(��(�C���                                    BxkW�  "          @�
=@0  ���@p�B�C��@0  �2�\?��A�G�C�AH                                    BxkW+�  T          @�
=@Mp���@z=qB6�C�}q@Mp��W
=@9��A��C��R                                    BxkW:6  �          @�z�@h���vff?�\)A�(�C��{@h����>�?��C���                                    BxkWH�  T          @�Q�@P���mp�?޸RA�
=C���@P�����\>�z�@I��C�K�                                    BxkWW�  T          @�{@W
=���R?�(�A�p�C�H�@W
=��G�=�Q�?n{C�9�                                    BxkWf(  
�          @�(�?�G��q�@-p�Bz�C��)?�G���  ?�A^=qC�AH                                    BxkWt�  
�          @�G����
?��@�{B���C!H���
�333@���B�\CL��                                    BxkW�t  T          @���<��?���@�BO�Ck��<��=�G�@�G�BiG�C1�=                                    BxkW�  T          @�\)>\)�z�@��RB���C��
>\)���@��B{  C���                                    BxkW��  
�          @�  <���
=@��B�G�C��q<��E@��\BeQ�C�G�                                    BxkW�f  �          @�z�>B�\��@�G�B��=C��3>B�\�e@�G�BH(�C���                                    BxkW�  T          @�G�>��H�7�@��Bn�
C��=>��H���R@i��B��C��                                    BxkW̲  	`          @�G�?����Q�@l��B#  C���?����G�?��HA��C���                                    BxkW�X  
�          @��R?h���3�
@���Bn�\C�H?h�����@n�RB�C�ٚ                                    BxkW��  �          @���?�����R@��Bk��C��?����tz�@s�
B&��C���                                    BxkW��  �          @��?��ÿ���@��B�ǮC�� ?����N{@��BW�RC�*=                                    BxkXJ  
�          @���?�p��dz�@�ffBFz�C��f?�p����@:=qA�33C�\                                    BxkX�  "          @��H?�
=���@S�
BQ�C��q?�
=���?��
AvffC�}q                                    BxkX$�  �          @��R@L�Ϳ���@���B=�C���@L���I��@HQ�B��C���                                    BxkX3<  �          @��@dz�W
=@�
=BJ��C�c�@dz��@y��B)�C�XR                                    BxkXA�  "          @���@��
��33@8Q�A�
=C��@��
�#33@�A�(�C��                                    BxkXP�  �          @�G�@|(��J=q@�A�p�C�� @|(��s�
?�
=AB=qC��)                                    BxkX_.  "          @�
=@]p��e�@$z�A�z�C��@]p���  ?�A@��C��{                                    BxkXm�  �          @�G�@l(��У�@Dz�B(�C�{@l(��%@�
AمC�s3                                    BxkX|z  �          @��@3�
�y��@6ffA��C��q@3�
��z�?���A[\)C��3                                    BxkX�   T          @��H@E�-p�@j�HB'{C�]q@E�s�
@ ��A��HC���                                    BxkX��  �          @�{@�\�}p�@1G�A���C��@�\��p�?��RAV=qC��                                    BxkX�l  T          @�?�G���{@7�B�C��?�G���p�?�  A[
=C��=                                    BxkX�  �          @�p�?����\��@�p�B?�C�:�?�����p�@/\)A�\C�G�                                    BxkXŸ  
�          @���?�Q����@��HB{�C�n?�Q��W
=@z�HB8�
C�O\                                    BxkX�^  �          @���?!G��>{@�
=BY
=C���?!G����@<��B
Q�C�9�                                    BxkX�  �          @�
=���H���\@��\B�\)CV�R���H��@n{BRCo�                                    BxkX�  �          @�
=��
=��ff@N�RBaG�Cl����
=�2�\@�HB��Cw                                    BxkY P  �          @��R@333>\)@��Bk�@<(�@333����@�BZ\)C�S3                                    BxkY�  �          @��
@AG����@�\)BX�C�)@AG�����@uB=�C�w
                                    BxkY�  "          @��\@W�>��@n�RB?33@��@W���G�@fffB6�C���                                    BxkY,B  �          @�G�@8��?=p�@Tz�B@33Afff@8�þ�
=@XQ�BD�C��
                                    BxkY:�  �          @�z�@��@1�@o\)B6ffBEz�@��?�{@��RBi�\A���                                    BxkYI�  
�          @���@��?���@
=A�  A�  @��?�@-p�A�p�@���                                    BxkYX4  �          @���@��׾aG�@>{B
��C���@��׿���@.{A�=qC�'�                                    BxkYf�  �          @��@��;.{?�\A�
=C��@��Ϳ@  ?�{A�=qC��                                     BxkYu�  �          @�(�@���E�?�
=A�=qC���@����\)?���A���C�0�                                    BxkY�&  �          @�
=@��\>�Q�?���A�@�  @��\���
?��HA�ffC�7
                                    BxkY��  T          @�z�@��׿�G�?���A��\C�T{@��׿�(�?�33AEC��3                                    BxkY�r  �          @�(�@�G���{@��A�{C���@�G���\)@��A��\C�p�                                    BxkY�  �          @��R@��k�@  Aď\C���@��xQ�@33A���C�n                                    BxkY��  �          @��@��Ϳ���@   A�33C�  @��Ϳ��?��RA\)C���                                    BxkY�d  "          @�(�@�{�@  @5�B C�33@�{��33@�A�
=C��{                                    BxkY�
  
�          @��H@s33�Q�@_\)B'z�C���@s33��
=@AG�BQ�C��3                                    BxkY�  "          @�(�@l���@QG�B\)C�C�@l���C33@��AԸRC�=q                                    BxkY�V  �          @��@tz���
@$z�A�p�C�l�@tz��AG�?�A��C��\                                    BxkZ�  �          @�
=@|(��u@S�
B�HC�  @|(��   @3�
B�HC��                                    BxkZ�  �          @�@��5�?�z�A���C���@��N{?#�
@�p�C�33                                    BxkZ%H  "          @�ff@�
=�5?ٙ�A��C�H@�
=�O\)?.{@�\)C�9�                                    BxkZ3�  "          @�33@�(��<(�?�=qA�z�C�Ff@�(��S33?�@���C���                                    BxkZB�  T          @��H@���P  ?�
=A�p�C��{@���l��?E�@��
C��                                    BxkZQ:  "          @��@��ff@1�A�C��@��8��?���A�  C�'�                                    BxkZ_�  �          @�33@�G���@@  A��
C���@�G��<(�@
=qA�=qC��                                    BxkZn�  �          @���@�녿�(�@_\)B�RC��@���0��@/\)A�p�C��\                                    BxkZ},  �          @�@�  �^�R@|��B)\)C�/\@�  �ff@]p�BffC��3                                    BxkZ��  �          @���@����p�@��B-�C���@���\)@`  B=qC��\                                    BxkZ�x  �          @��H@tz῀  @�(�BF
=C��=@tz��(�@��B'\)C��q                                    BxkZ�  �          @��\@��
>8Q�@p  B"��@=q@��
���
@g
=BffC�Y�                                    BxkZ��  
�          @��@�  >�z�@1G�A�@`  @�  �(�@.{A�z�C�S3                                    BxkZ�j  �          @�  @�?W
=@  A�A\)@�=��
@��A���?Tz�                                    BxkZ�  �          @�=q@��^�R@
=A��C�8R@���=q?��HA��C���                                    BxkZ�  
�          @��R@\(��(Q�@h��B G�C�O\@\(��i��@'
=A�{C��                                    BxkZ�\  �          @�  @z=q��
=@r�\B/�
C��=@z=q��{@]p�B=qC��3                                    Bxk[  T          @���@�33@<��?Q�AQ�A���@�33@"�\?�(�A��A���                                    Bxk[�  T          @���@�{=u@A��?!G�@�{�(�@   A��C���                                    Bxk[N  �          @��\@�
=����?�\)Ad��C��H@�
=��Q�?aG�A=qC�f                                    Bxk[,�  �          @�{@�
=�=q?���A��HC�|)@�
=�2�\?B�\@�C���                                    Bxk[;�  "          @���@�=q�G�?��
A2ffC���@�=q�\)>�{@k�C��=                                    Bxk[J@  T          @��R@���G�?}p�A*�RC�.@���p�>\@��
C�                                    Bxk[X�  �          @��
@��׿�(�?xQ�A)p�C��@�����>���@O\)C���                                    Bxk[g�  �          @��R@�33�&ff�7���RC�*=@�33>�\)�;���ff@QG�                                    Bxk[v2  "          @�(�@�(��B�\�QG���HC��f@�(�?Y���J�H�p�A.�R                                    Bxk[��  "          @�=q@�  =����G��̸R?�G�@�  ?O\)�Q����RA�R                                    Bxk[�~  T          @��H@�33?8Q��1�����A�@�33?�ff�=q�ә�A�p�                                    Bxk[�$  �          @�ff@�33���Q���=qC��\@�33?5�G�����@��                                    Bxk[��  
�          @�  @�
=?=p��ff���Ap�@�
=?�{��\��33AiG�                                    Bxk[�p  �          @�p�@��R?��ý��Ϳ���A��
@��R?�G�>�@��\A�G�                                    Bxk[�  
�          @��
@��
@��Tz��{A�Q�@��
@G�����
=A�=q                                    Bxk[ܼ  
(          @�  @�ff��@(��A�z�C�K�@�ff�n{@{A�33C���                                    Bxk[�b  �          @�Q�@L�Ϳ&ff@��BZ��C�L�@L����@�B?(�C�˅                                    Bxk[�  
�          @��@dz�=�G�@�
=BMz�?�Q�@dzῢ�\@���BB  C�/\                                    Bxk\�  
�          @�=q@J=q>�33@�=qBbp�@�33@J=q���@�ffBY�RC��                                    Bxk\T  
�          @���@���>���\)�6ff@�{@���?������
@��H                                    Bxk\%�  U          @�ff@�p����?�@�  C��)@�p����>�
=@�z�C���                                    Bxk\4�  !          @���@x���7
=@333A���C�ٚ@x���e�?�ffA�(�C��{                                    Bxk\CF  �          @��@fff�J=q@p�A�Q�C�` @fff�l(�?�33AHz�C�/\                                    Bxk\Q�  �          @��@�{���?��\A*�RC�J=@�{�ff>��
@S�
C�K�                                    Bxk\`�  �          @��
@Dz��XQ�@Dz�B�C��@Dz����?�33A���C�<)                                    Bxk\o8  �          @���@%�
=@�=qBW�C���@%�mp�@xQ�B"ffC�n                                    Bxk\}�  
�          @���@#�
��Q�@fffBz�C�C�@#�
��p�@{A��RC��H                                    Bxk\��  T          @���?�z�����@1G�A�C�.?�z����?�Q�A@Q�C�K�                                    Bxk\�*  T          @�Q�?�\)����@G�A��C�(�?�\)���?0��@�  C�n                                    Bxk\��  �          @�33?�����@G�B(�C�G�?����G�?�{A\)C��)                                    Bxk\�v  �          @��\@z����\@333A��C�%@z����R?��AO�
C��3                                    Bxk\�  T          @���@����=q?��A��
C�B�@�����>�33@aG�C�}q                                    Bxk\��  �          @�33@\)��33?#�
@�\)C�5�@\)���\�B�\�G�C�B�                                    Bxk\�h  �          @�ff@Mp��j�H?���AJ�HC��3@Mp��u��L�Ϳ��C��R                                    Bxk\�  �          @�ff?�(��|��@��\B1��C�H�?�(����@=p�A���C���                                    Bxk]�  
�          @�(�@��{�@���B+��C��@����@2�\Aڏ\C�P�                                    Bxk]Z  
�          @Å?�{��ff@��HB+\)C��
?�{��
=@*=qA�{C�J=                                    Bxk]   �          @��@ �����\@Y��B�HC���@ ����(�?��A��C�:�                                    Bxk]-�  �          @�ff@���p�@   A�(�C�J=@�����>��R@7
=C��R                                    Bxk]<L  
�          @��?�{����?c�
A�\C���?�{�����\(��{C��)                                    Bxk]J�  �          @�z�Q���33@�ffB-��C���Q���z�@.�RA��HC��=                                    Bxk]Y�  �          @������=q@��B,=qC�
�����33@-p�A��C�>�                                    Bxk]h>  	�          @Å������ff@=p�A�RC��������H?�ffAEC���                                    Bxk]v�  
�          @��
��ff�<(�@�ffBp�HC�����ff���\@��B-Q�C�H                                    Bxk]��  
�          @��H�.{���@�G�B���C~@ �.{�z=q@�33BD�HC��                                    Bxk]�0  T          @�33���
�u@�{BJ�C��3���
��G�@W�B�C��q                                    Bxk]��  
�          @�33�����z�@�ffB<\)C��þ����  @C�
A�\)C�.                                    Bxk]�|  T          @�=q?�Q��i��@���BD�
C��?�Q���=q@R�\B�HC�T{                                    Bxk]�"  �          @�Q�>������@7�A��C�Ф>������?�\)Ab�RC�}q                                    Bxk]��  �          @����#�
���@k�B�C���#�
����@
�HA�33C�/\                                    Bxk]�n  "          @�G�>�Q���33@��\B9C��=>�Q���@=p�A�C��                                    Bxk]�  "          @����!G���{@   A�G�C�^��!G���ff?xQ�A Q�C���                                    Bxk]��  	�          @��R@���33����-p�C�� @���녿�(�����C�w
                                    Bxk^	`  
�          @�z�?���G�������C��?���
=�^�R��\C��q                                    Bxk^  "          @�ff>����ff�c33�
=C��>���_\)��  �V�C�@                                     Bxk^&�  
�          @���?�  ����9�����C��\?�  �����  �5��C��                                    Bxk^5R  �          @��@A���ff����|Q�C��q@A������>�R��\)C���                                    Bxk^C�  T          @�Q�?�z�����0���أ�C��?�z��������z�C���                                    Bxk^R�  T          @���?��H��
=��������C�u�?��H��
=�U��{C���                                    Bxk^aD  T          @��H@p���p�?0��@��
C�33@p���z�aG��
=C�@                                     Bxk^o�  U          @���@
=q���R>�{@N�RC�` @
=q���\����B�HC��)                                    Bxk^~�  !          @�z�?Q���������s�
C��?Q���(������HC�#�                                    Bxk^�6  T          @��
@z����\>aG�@�C�K�@z���p���\)�Q�C���                                    Bxk^��  "          @�ff?����(�>�z�@.{C�u�?����
=�����O\)C���                                    Bxk^��  "          @�\)?޸R���?8Q�@ָRC�0�?޸R���
�xQ��  C�<)                                    Bxk^�(  
�          @�
=?�{��ff�����C33C��{?�{���
���R��Q�C�H                                    Bxk^��  
Z          @�\)@����33>�33@R�\C���@����\)���R�:=qC�\                                    Bxk^�t  "          @�
=@R�\�5�@��B4�\C���@R�\�|(�@Y��B�HC��3                                    Bxk^�  "          @�(�@����
@>�RA�C�
=@���Q�?�  Ac
=C�Ǯ                                    Bxk^��  T          @�(�@����G�@
=A��\C��@����p�?��@�ffC�W
                                    Bxk_f  T          @���@\)����?#�
@��C�˅@\)����u�  C��                                     Bxk_  �          @�z�@�R��녽�G���G�C�H@�R��녿�33�{�
C��                                     Bxk_�  �          @\@����
=?�@���C��@����p��z�H�G�C�*=                                    Bxk_.X  �          @��
@C�
���?�z�A0��C�xR@C�
��\)�����5�C�,�                                    Bxk_<�  �          @�\)@C�
��33>�33@N{C�� @C�
�����33�+�C�#�                                    Bxk_K�  �          @�ff@<����\)@{A��C���@<�����?Q�@���C��f                                    Bxk_ZJ  S          @ȣ�@-p���p�@-p�Aܣ�C��)@-p���  ?���AZ�RC�8R                                    Bxk_h�  "          @�  @5��@��B\C��@5��Y��@�G�B0ffC��                                    Bxk_w�  
�          @���@33�
=@�  Bl�
C�s3@33�mp�@�z�B8{C�n                                    Bxk_�<  
�          @��?ٙ��l(�@���BA{C�U�?ٙ�����@W�B�HC���                                    Bxk_��  "          @���@
�H��p�@p�A�p�C�c�@
�H��z�?s33A�\C���                                    Bxk_��  
�          @�=q@/\)���H?�\)Ay��C�(�@/\)���H>#�
?\C���                                    Bxk_�.  S          @���@:=q��  ?޸RA���C�
@:=q����>���@8��C�h�                                    Bxk_��  
�          @��R@QG��Z�H@C�
B��C�޸@QG���(�@ ��A�p�C�5�                                    Bxk_�z  T          @�(�@�(��n{?�  AG�C���@�(��vff�L�Ϳ�C�XR                                    Bxk_�   �          @�{@����C�
?���A;�C�@����QG�>��
@H��C�&f                                    Bxk_��  �          @�{@�p��Q�?�A�{C��f@�p��#33?���A;�C��R                                    Bxk_�l  �          @���@�{��\)�
=q����C���@�{�:�H������\C�,�                                    Bxk`
  �          @��R@���>��������@��
@���?aG������d  A�                                    Bxk`�  
�          @�G�@�{�   � ����G�C�ff@�{>L���#33�ȸR@ff                                    Bxk`'^  
�          @�G�@�Q�#�
�333��Q�C���@�Q�?8Q��-p���(�@�G�                                    Bxk`6  �          @�=q@��>�{�>�R��R@p  @��?����1���33AC33                                    Bxk`D�  T          @�33@�33>��
�3�
�ݙ�@^{@�33?����'���p�A4��                                    Bxk`SP  "          @�33@�z�?k��!G��ř�Aff@�z�?Ǯ�����A���                                    Bxk`a�  �          @��@��?�G��Q����\A�Q�@��@�������33A��
                                    Bxk`p�  T          @�  @��?��Ϳ�{��z�A�
=@��@녿�ff�H(�A��H                                    Bxk`B  T          @�  @�?�=q��Q��]G�AS33@�?����
�A��H                                    Bxk`��  
�          @�
=@�G�?!G������@�  @�G�?�z�����
A=                                    Bxk`��  �          @���@�33>���Q��7�@�
=@�33?B�\���
�{@�(�                                    Bxk`�4  #          @���@��?0�׿�33�0��@�
=@��?xQ�n{�A�R                                    Bxk`��  !          @�  @��?!G������=q@��H@��?:�H��z��333@�\                                    Bxk`Ȁ  T          @�  @�p�?J=q��G�����@��@�p�?J=q=���?n{@�\                                    Bxk`�&  "          @���@�z�?�  ?.{@�
=A�\@�z�?J=q?k�Az�@��
                                    Bxk`��  �          @�@���?���?   @�33A<  @���?��
?L��@�Q�A!G�                                    Bxk`�r  
Z          @�z�@���?��Ϳ�33�4z�A0(�@���?�{�Q�� (�AX��                                    Bxka  "          @���@��
?Q녿��`Q�Az�@��
?�zΐ33�5A:�H                                    Bxka�  T          @�Q�@��R?:�H������@�p�@��R?���
=���RA\(�                                    Bxka d  
�          @��@�  ?�������
=ADz�@�  ?�z����p��A�ff                                    Bxka/
  T          @�p�@�z�?˅�.{��Q�A|Q�@�z�?�=q>�  @(�Az�R                                    Bxka=�  
�          @���@���>�ff�3�
��=q@��H@���?��&ff��AG�                                    BxkaLV  T          @�  @��?��
�%��˙�A�@��@���G�����Aȣ�                                    BxkaZ�  T          @��H@�{?�33�=p�����AG�
@�{?���#�
����A�p�                                    Bxkai�  "          @���@�ff?�
=�����\Ak
=@�ff@   ����(�A���                                    BxkaxH  "          @��@��?���
���@�Q�@��?�{�ff��  A733                                    Bxka��  	�          @���@�?+�������@�\)@�?�{�Ǯ�p��A1G�                                    Bxka��  
�          @���@�
=?�  ���R�@Q�A
=@�
=?���n{���AK33                                    Bxka�:  T          @�G�@�
=?�\)��=q�*  A�=q@�
=@Q��dz����A�G�                                    Bxka��  
�          @�G�@�z�>��H�Dz���(�@��@�z�?��\�5���A]�                                    Bxka��  
�          @�Q�@���>�{�q��(�@���@���?�ff�c�
�Az=q                                    Bxka�,  �          @�\)@�p�?���XQ���AH��@�p�?���@  ���\A��                                    Bxka��  �          @�p�@��?^�R�L����HA#\)@��?��7���A��                                    Bxka�x  �          @�@�p�?k��Z�H���A3
=@�p�?�\�Dz����\A�=q                                    Bxka�  T          @�G�?������@z�A�Q�C��?�����  ?�=qA6�\C�"�                                    Bxkb
�  
�          @�
=?�(�����@9��A�ffC��
?�(����?\AmG�C�(�                                    Bxkbj  
�          @���?��H���H@��A���C�� ?��H����?s33AQ�C�{                                    Bxkb(  
�          @��R@ff����?�ffAtQ�C�B�@ff���
>8Q�?޸RC��                                    Bxkb6�  T          @�\)@!���\)?�z�A[
=C��@!���p�<�>��
C��                                     BxkbE\  �          @���@.{��(�>�@�=qC��@.{���\�\(���C�\                                    BxkbT  
�          @��
@U���=q�G���z�C��\@U��j=q�C33��33C�!H                                    Bxkbb�  �          @���@Vff���
����Z�RC��q@Vff�x���=q����C�aH                                    BxkbqN  �          @�z�@�  �fff�������HC���@�  �AG��*�H��33C�S3                                    Bxkb�  
�          @���@����+��
=����C�` @�����
�.{���HC�^�                                    Bxkb��  "          @���@O\)�hQ쿫��v�RC��@O\)�J�H�{��Q�C��3                                    Bxkb�@  T          @���?xQ����Ϳ@  ���C�q?xQ���G����
=C�y�                                    Bxkb��  
�          @���@
=�����z�����C�  @
=�j�H�Vff���C�b�                                    Bxkb��  �          @�  @��H�y���z�H�G�C�+�@��H�a녿�
=��\)C���                                    Bxkb�2  $          @�\)@�33�U���\)��(�C�J=@�33�3�
�=q�Ǚ�C��\                                    Bxkb��  �          @��@�
=�c33��p��p��C��3@�
=�Dz�����ffC�                                      Bxkb�~  T          @�Q�@�=q�S�
�Ǯ�}p�C��@�=q�Fff���H�Ep�C��                                    Bxkb�$  �          @��@���dz�����RC��f@���Tzῳ33�b�HC���                                    Bxkc�  �          @��@tz��H���$z����
C�B�@tz�����Q��{C��                                    Bxkcp  T          @���@@���1��l���(Q�C�� @@�׿�\�����K(�C���                                    Bxkc!  
�          @�\)@aG��.{�~{�&C�%@aG���33�����E33C�y�                                    Bxkc/�  
�          @�\)@W��5������*Q�C�  @W���p���(��J��C�j=                                    Bxkc>b  �          @�  @\(��333�����)��C�k�@\(��ٙ����
�I\)C��
                                    BxkcM  
�          @�Q�@`���1��\)�&�
C�Ф@`�׿ٙ������E��C�
                                    Bxkc[�  T          @��@W
=�.�R���
�.\)C�u�@W
=��\)��p��M�C�#�                                    BxkcjT  "          @�
=@W��G���33�;��C��@W���\)�����U�C��\                                    Bxkcx�  T          @�
=@Mp���
��
=�BG�C�"�@Mp���\)����]33C�U�                                    Bxkc��  T          @�p�@vff�Q��A���
=C�Ф@vff��H�o\)��C��=                                    Bxkc�F  �          @��@2�\���?\)��  C��q@2�\�S�
�{��(��C��                                    Bxkc��  �          @��@>�R�����C�
���C�=q@>�R�I���}p��)��C���                                    Bxkc��  �          @�z�@W��a��S33�=qC��\@W��'
=��=q�.C�q                                    Bxkc�8  "          @��@>{�:=q��G��0C��@>{������(��T=qC�=q                                    Bxkc��  �          @�=q@&ff�=p������=�RC��=@&ff����(��d{C��3                                    Bxkc߄  "          @�
=@�\�c�
���\)C�@ @�\>����
=�=A5��                                    Bxkc�*  �          @�Q�@�ÿ������x�
C��@��=�����z�@{                                    Bxkc��  T          @�(�@)���\���u��C���@)��?W
=���
�q  A�z�                                    Bxkdv  
�          @���@K���(��c�
�.\)C�)@K�����|(��F�\C��R                                    Bxkd  �          @��R@\(��`  �9�����HC�<)@\(��+��j�H� ffC��                                    Bxkd(�  �          @�\)@j=q�����Ϳ���C��q@j=q���ÿ�\)�:ffC�%                                    Bxkd7h  �          @��H@^�R�;��N�R��C��3@^�R�33�vff�.��C���                                    BxkdF  �          @��\@�������
�̏\C��
@��׿���333���
C�y�                                    BxkdT�  T          @�(�@�ff�N{�@  ���C�
=@�ff�<�Ϳ�G��|��C�AH                                    BxkdcZ  T          @�p�@�z��7
=�Ǯ��  C�(�@�z�����p���C�W
                                    Bxkdr   �          @�
=@�Q��<(���=q��ffC�z�@�Q�����\)���C���                                    Bxkd��  
�          @���@�  �X�ÿ\�z�\C��\@�  �:�H�33���\C��q                                    Bxkd�L  T          @�
=@����`�׿�ff��33C���@����A��
=��  C��R                                    Bxkd��  �          @��@����o\)��
=�AC���@����U�33��33C�*=                                    Bxkd��  �          @��R@vff����Ǯ�\)C���@vff�x�ÿ����aC�]q                                    Bxkd�>  T          @�@s�
�~{>���@�(�C��@s�
�|�Ϳz���  C��)                                    Bxkd��  T          @�@Z=q��z�?xQ�A�C���@Z=q��  ����G�C��
                                    Bxkd؊  T          @�
=@S33��p�?���A.�HC�^�@S33�����#�
��Q�C��)                                    Bxkd�0  T          @�
=@c�
�g�@#�
A�  C�@ @c�
��(�?�{A�p�C�]q                                    Bxkd��  
�          @�p�@dz��c�
@#�
A�G�C���@dz���=q?�\)A��C��)                                    Bxke|  $          @��R@Q��|��@(�A˙�C�ٚ@Q���?�Ag\)C�G�                                    Bxke"  �          @�p�@�G��Vff����\)C��)@�G��P  �Tz����C�l�                                    Bxke!�  "          @���@\)�Tz�=�Q�?��C��@\)�O\)�5���\C�p�                                    Bxke0n  �          @��@ ����Q�@  A�=qC���@ ����p�?���A4  C���                                    Bxke?  �          @���@(Q�����?��@ǮC�t{@(Q����׿���{C�q�                                    BxkeM�  �          @��
@=p����@
=A�G�C�@=p���  ?���A[�C��f                                    Bxke\`  �          @�z�@�33�h��?�33A>ffC�7
@�33�s�
>�  @#33C��\                                    Bxkek  �          @�  @�{�dz�>���@Tz�C��3@�{�c33�
=q��z�C���                                    Bxkey�  "          @��R@�p��*=q��G��#33C���@�p����{���C�K�                                    Bxke�R  �          @�33@�  ��H�޸R��  C���@�  ��Q��G���ffC�Z�                                    Bxke��  �          @�Q�@p  �Vff��ff���HC�  @p  �8Q��33�ϮC�<)                                    Bxke��  �          @���@�(��Fff?�ffAZ�\C�Z�@�(��Tz�?�@�
=C�j=                                    Bxke�D  �          @���@�Q��/\)>�?�{C��@�Q��,�Ϳ���Q�C�:�                                    Bxke��  T          @���@�\)�8�ÿ�{���C���@�\)��H�  �ƸRC�)                                    Bxkeѐ  �          @��@w��\)�<(��\)C���@w���p��X���G�C���                                    Bxke�6  "          @�p�@:�H�/\)�n{�R=qC�n@:�H�(��Ǯ��Q�C�
                                    Bxke��  �          @�G�=�G���@���Bqz�C�C�=�G��XQ�@e�B:z�C���                                    Bxke��  �          @�p�=u�#33@u�Bap�C���=u�X��@G�B*p�C��f                                    Bxkf(  T          @�G���녿޸R@��\B�k�Cl������9��@���B]Cxp�                                    Bxkf�  �          @�  =�G��
=@�G�B�� C�q�=�G��O\)@�z�BV�C��                                    Bxkf)t  �          @�(�>B�\�J�H@�p�BR�HC���>B�\��=q@S33B{C�Q�                                    Bxkf8  "          @������R@��BkffCi8R���P��@|��B<��Cr��                                    BxkfF�  "          @��R��{�7
=@�z�BX�
Ct�)��{�tz�@eB&{Czn                                    BxkfUf  "          @�{@@  �XQ�@5B ��C��f@@  �}p�?��HA�=qC���                                    Bxkfd  
�          @��@|(��<(�@�A��C���@|(��U?�=qAf=qC��
                                    Bxkfr�  "          @��R@j=q�W
=@��A�  C���@j=q�qG�?��
A[33C��                                    Bxkf�X  �          @�{@K�����=�G�?�
=C�\)@K���p��fff�  C���                                    Bxkf��  �          @�z�@ ���������\)C��
@ ����33��=q���C��)                                    Bxkf��  T          @�(�@�p��N{�������
C�(�@�p��A녿�33�Ip�C���                                    Bxkf�J  �          @�z�@�
=�7�?:�H@��C���@�
=�=p�<�>���C�:�                                    Bxkf��  �          @�p�@�  �(��?��HAP(�C��@�  �6ff?��@�(�C��{                                    Bxkfʖ  �          @�p�@���K�=��
?\(�C�˅@���Fff�+���{C��                                    Bxkf�<  �          @�z�@��
���>B�\@ ��C��R@��
����{�fffC��\                                    Bxkf��  �          @�G�@�\)�)��?(��@�\C���@�\)�.�R<��
>��C���                                    Bxkf��  �          @��\@�녿�=q?��A9��C���@����?��@У�C��3                                    Bxkg.  �          @��
@�����ff�W
=�p�C���@����Ǯ���R�R�\C��                                    Bxkg�  �          @�G�@��H>��Ϳ�R��@��@��H?���\��@��                                    Bxkg"z  �          @��@��?@  �:=q�
�A$(�@��?��)������A�Q�                                    Bxkg1   "          @��
@i��?(���`  �-(�A$z�@i��?�(��O\)�  A��H                                    Bxkg?�  
�          @���@r�\>�G��R�\�#Q�@��
@r�\?����Fff��
A��R                                    BxkgNl  �          @�(�@~�R���E�z�C�L�@~�R>aG��HQ��z�@Mp�                                    Bxkg]  �          @�z�@�\)�W
=�=q��C��f@�\)>�Q������  @��H                                    Bxkgk�  �          @��@�
=�
=��=q�C33C��R@�
=���
��Q��W33C�#�                                    Bxkgz^  �          @�p�@��>8Q�?=p�A��@�
@��    ?B�\A��=#�
                                    Bxkg�  �          @��
@��׿   ?��HA���C��)@��׿c�
?��A�
=C��
                                    Bxkg��  �          @�ff@p��=���@e�B.33?��
@p�׿=p�@`  B)��C�p�                                    Bxkg�P  T          @�p�@hQ�?fff@b�\B-�A]��@hQ�=#�
@i��B4��?+�                                    Bxkg��  �          @�p�@��?��H@�
AٮA�z�@��?�\)@)��A��\Ap��                                    BxkgÜ  T          @��@���?�\)?�A�G�A��@���?�
=@   A�G�Ai                                    Bxkg�B  �          @��@��>�
=��{�W33@�ff@��?+��z�H�=A�                                    Bxkg��  
�          @�{@�G�>��G
=��\@��H@�G�?����:=q�
�A�{                                    Bxkg�  �          @�  @�  >����H���@�\)@�  ?J=q����h��A
=                                    Bxkg�4  T          @���@��\?���u�(  A;�
@��\?�  �0����  A]p�                                    Bxkh�  �          @���@��?��Ϳ�  �/\)AD(�@��?�ff�8Q����\Ag33                                    Bxkh�  �          @��R@��?@  �(����  Aff@��?c�
����33A��                                    Bxkh*&  �          @�\)@�{>��
��G����@c33@�{>�
=��33�vff@�=q                                    Bxkh8�  �          @��@�\)�aG�=��
?fffC��f@�\)�u<�>���C��{                                    BxkhGr  "          @�
=@����=�Q�?�  C�g�@������
�uC�\)                                    BxkhV  T          @�ff@���>�>���@�\)?�Q�@���<�>�
=@�{>�Q�                                    Bxkhd�  �          @�{@�\)����?�ffA��C�1�@�\)��33?���AV�RC�h�                                    Bxkhsd  
�          @���@��
<��
?s33A1�>8Q�@��
�W
=?n{A,��C�Ǯ                                    Bxkh�
  T          @��@�����ff?�{A|z�C�P�@����B�\?�(�Aap�C��                                    Bxkh��  �          @��H@���   ?�(�Ab{C��@����R?8Q�A��C��R                                    Bxkh�V  �          @��@�(��"�\>�(�@��
C��@�(��$z�.{����C��=                                    Bxkh��  �          @��@��
�:�H?:�HA�
C�Z�@��
�@��<�>�33C���                                    Bxkh��  �          @��@�
=�-p�>�{@vffC��=@�
=�-p������\��C��                                    Bxkh�H  �          @���@�p��2�\�����RC�
@�p��-p��.{��  C�}q                                    Bxkh��  T          @�
=@���fff��33��Q�C���@���C33�,(��
��C�                                      Bxkh�  �          @�=q@�(��k��k��)��C��@�(��W
=��G����C�R                                    Bxkh�:  �          @�33@��\��(����
�n{C���@��\��z��
=����C���                                    Bxki�  T          @�33@w
=�G�?�G�A8(�C��f@w
=�QG�>��@9��C��q                                    Bxki�  �          @�G�@��\����?^�RA�HC�@��\�   >��@�ffC�8R                                    Bxki#,  
�          @�G�@��H���H>8Q�@33C��)@��H��(���G����RC��                                    Bxki1�  
�          @���@�ff�#�
���
�z�HC�˅@�ff�#�
��Q쿆ffC��=                                    Bxki@x  
�          @��@��H?�zΐ33�^�\A���@��H?�녿:�H���A�                                    BxkiO  
Z          @��R@�  ?=p�?�@���A{@�  ?��?0��A{@�\)                                    Bxki]�  �          @��@�
=?z�@ffA�@���@�
==�\)@�A�  ?xQ�                                    Bxkilj  �          @��@�@��׿�z����B{�H@�@�\)�\���\B�#�                                    Bxki{  T          @��@e@AG���G��C
=B   @e@J�H��\)�X��B%��                                    Bxki��  
�          @�ff@{�@��@�A�\C�9�@{�Z�H?�{A��
C��H                                    Bxki�\  T          @�@J�H�K�@p�A��C�u�@J�H�j�H?�z�A���C�n                                    Bxki�  �          @�{@W
=�(��@7
=B33C��@W
=�O\)@
�HA�p�C��                                    Bxki��  �          @��@C33�W�?�z�A��\C�
@C33�n{?�=qAI��C��                                    Bxki�N  �          @���@�{��  ?�33A��C��@�{����?��Am�C���                                    Bxki��  �          @���@��R���?�p�AW�C���@��R�Ǯ?c�
A33C�AH                                    Bxki�  T          @�{@�
=���?���Am��C�J=@�
=�E�?�
=AR�RC��q                                    Bxki�@  �          @���@�{>.{?��AP(�?���@�{����?�33AQ��C�g�                                    Bxki��  �          @��R@���?#�
?�=qA@z�@�
=@���>�p�?���AU�@�{                                    Bxkj�  �          @��\@�z��?��\AS\)C��)@�z�Y��?�{A733C��                                    Bxkj2  �          @�\)@��׿�=q?��A*=qC�z�@��׿��?G�@���C�g�                                    Bxkj*�  �          @���@�p��B�\?c�
Az�C��@�p��p��?0��@�{C��                                    Bxkj9~  �          @�33@�Q�.{?.{@���C�z�@�Q�Q�?�\@�ffC�                                    BxkjH$  �          @��
@��þ���?n{A�C�G�@��ÿ
=q?Q�A�RC�7
                                    BxkjV�  �          @��@��R    ?��
A+�=L��@��R�u?�G�A'
=C��H                                    Bxkjep  �          @���@�ff>\?s33A�R@}p�@�ff>\)?��\A(��?�G�                                    Bxkjt  �          @�ff@��>���?���A?�@Q�@��<�?�AF�\>�33                                    Bxkj��  �          @��@�=q?aG�?E�A�HA{@�=q?+�?uA"{@�
=                                    Bxkj�b  T          @��\@���G�?޸RA�G�C�Ǯ@����z�?�G�A{33C���                                    Bxkj�  �          @�z�@�  ��G�?s33A  C���@�  �&ff?Q�A�\C��H                                    Bxkj��  �          @��@��R�L��>��@�33C��q@��R���>Ǯ@w�C�C�                                    Bxkj�T  �          @�  @�ff���>�G�@��C�@ @�ff�#�
>���@<��C���                                    Bxkj��  T          @�z�@�
=��  ?Q�A�\C��3@�
=��z�?�@���C���                                    Bxkjڠ  T          @��H@�G����>���@��HC��@�G���{>���@W
=C�<)                                    Bxkj�F  �          @���@��R�0��>\@z=qC�l�@��R�@  >W
=@Q�C��                                    Bxkj��  �          @��H@����&ff=�G�?�=qC���@����&ff�L�Ϳ   C���                                    Bxkk�  �          @��
@��\�!G���G�����C�˅@��\�zᾀ  �"�\C��                                    Bxkk8  
�          @��H@����\)>��@*�HC�  @������=�?�  C��                                    Bxkk#�  �          @��@�녿333=L��>��C�c�@�녿333����p�C�o\                                    Bxkk2�  T          @�33@�=q��녾\�z=qC��H@�=q���R�����G�C�ff                                    BxkkA*  �          @�33@�녾�
=�����Dz�C��
@�녾�{�Ǯ��  C�B�                                    BxkkO�  �          @���@��\�G�=#�
>�Q�C�f@��\�E������G�C�R                                    Bxkk^v  �          @��@��
�(��������C��f@��
�\)��z��>�RC�*=                                    Bxkkm  T          @���@����;�ff��p�C�"�@����(�����z�C���                                    Bxkk{�  �          @���@�{�\(������\��C���@�{�@  ����{C�                                    Bxkk�h  �          @�33@���>�33��(����R@j=q@���>�ff��{�^�R@�33                                    Bxkk�  T          @���@�(�?k��.{��A(�@�(�?����G���(�A2=q                                    Bxkk��  
�          @���@��>�Q쿎{�?33@z=q@��?(���  �+
=@�z�                                    Bxkk�Z  T          @�
=@|�Ϳ+��"�\�\)C�5�@|�ͽL���'��Q�C���                                    Bxkk�   �          @��@�
=>��Ϳ�\)�J=q@��
@�
=?&ff�}p��333@�                                    BxkkӦ  "          @�z�@����\����G
=C�Q�@�녿�
=�:�H��C��R                                    Bxkk�L  T          @�{@����{�fff�%G�C�� @��Ϳ�Q쿱���p�C��                                    Bxkk��  �          @�@�p��p��O\)��
C��H@�p���Ϳ�\)��G�C�q                                    Bxkk��  
�          @��\@�����z�B�\�z�C�1�@����s33���\�;�C�g�                                    Bxkl>  
�          @��@�������\)�Tz�C�!H@���}p����
�c�
C�c�                                    Bxkl�  �          @��@�Q����?!G�@�Q�C���@�Q��z�>8Q�@ ��C��                                    Bxkl+�  
Z          @��@�33��>Ǯ@�G�C�Ф@�33�#�
>u@*�HC�ff                                    Bxkl:0  T          @��@����   >�G�@�ffC�1�@����
=>��R@^�RC���                                    BxklH�  �          @��@�\)�z�?�@���C��\@�\)�0��>\@���C��                                    BxklW|  "          @��\@�  �Tz�>B�\@�C�Ff@�  �Y���#�
����C�'�                                    Bxklf"  
�          @��R@������?J=qA�C���@����(�>���@�33C�˅                                    Bxklt�  �          @�@�녿fff?   @���C���@�녿�  >�=q@HQ�C�&f                                    Bxkl�n  �          @�
=@�p����R?�\@�33C�7
@�p���
=>�
=@��HC��3                                    Bxkl�  T          @�ff@�(�?��?\)@ӅA��@�(�?�{?k�A+�
A�p�                                    Bxkl��  "          @��@�  ?�  �:�H�	A�@�  ?�\)�����]p�A��\                                    Bxkl�`  T          @�p�@��\���Tz���C��\@��\>.{�O\)�z�?��H                                    Bxkl�  �          @���@��
�L�;��
�s�
C��@��
����Q�����C�K�                                    Bxkl̬  �          @�33@�p�>�\)>�@��@Z=q@�p�>#�
?�@ə�?��H                                    Bxkl�R  T          @�33@��>�  @G�AƏ\@R�\@���u@G�AƸRC�p�                                    Bxkl��  �          @�z�@�
=?�@z�A�
=@�p�@�
=�#�
@Q�A�p�C��q                                    Bxkl��  �          @�33@�z�>B�\@{A�{@#�
@�z���@(�A�
=C�.                                    BxkmD  "          @��
@�
=>�
=?���A���@�33@�
==u?�G�A��H?G�                                    Bxkm�  
�          @���@w�>���@��A���@���@w��B�\@�B Q�C���                                    Bxkm$�  "          @��@�Q�=�\)@�A���?n{@�Q��@�RAۅC��                                    Bxkm36  �          @���@���Q�?�p�A�C��{@�����H?�p�A�G�C�w
                                    BxkmA�  T          @�=q@�\)���?�z�A�p�C��
@�\)����?�=qA�33C�Z�                                    BxkmP�  �          @��\@��\��(�?�p�A��
C�.@��\���?�G�A�z�C��
                                    Bxkm_(  �          @�p�@�\)    >���@i��C��q@�\)��\)>�z�@b�\C��                                    Bxkmm�  �          @���@��R?�G��@  �
=qA��R@��R?�33��p���{A���                                    Bxkm|t  �          @��H@���@#�
�.{��  A�G�@���@(�ý#�
��A�(�                                    Bxkm�  �          @�Q�@z�H@8��>�
=@�z�B��@z�H@-p�?���APQ�B
��                                    Bxkm��  �          @�@��?}p����\�@��A@(�@��?��H�@  �p�AhQ�                                    Bxkm�f  T          @���@�z᾽p���{�~{C���@�zᾊ=q��
=��(�C�e                                    Bxkm�  �          @�z�@��\���H?&ff@�{C�,�@��\����>�\)@Q�C���                                    BxkmŲ  
�          @�@u��1G������\C�@u��*=q�O\)�{C���                                    Bxkm�X  
Z          @�=q@�E@*�HB
=qC���@�i��?���A�ffC�Y�                                    Bxkm��  
�          @�{?�\)���
@p�A�=qC�/\?�\)��33?��AzffC�\                                    Bxkm�  �          @�  ?�����R?ٙ�A�
=C���?����\)?�\@��C���                                    Bxkn J  
+          @��\@#33���
?
=@���C�޸@#33��z��(����C��\                                    Bxkn�  �          @�Q�@]p��O\)?ǮA���C�k�@]p��a�?8Q�A��C�:�                                    Bxkn�  T          @��@j=q�!G�?��A�ffC��q@j=q�9��?�(�Aj�\C���                                    Bxkn,<  T          @�p�@���HQ�@+�B�C�*=@���l(�?�A��RC��                                    Bxkn:�  T          @��H@\(��X��@
=A���C���@\(��s33?�(�AVffC��                                    BxknI�  �          @��@s�
�b�\?(��@�C���@s�
�e�k���RC�`                                     BxknX.  �          @��\@r�\�fff?G�A(�C�=q@r�\�k�����ffC��                                    Bxknf�  T          @��@Vff�Z�H?O\)AC�0�@Vff�aG��L�Ϳ
=C���                                    Bxknuz  �          @�z�@w
=�P��?L��A  C���@w
=�Vff���
�k�C���                                    Bxkn�   �          @�
=@�33�+�?h��A"{C�(�@�33�5�>k�@!G�C�~�                                    Bxkn��  �          @�ff@�{���
?h��A!p�C��)@�{���H?�@���C���                                    Bxkn�l  �          @�=q@�G�>������
=@G�@�G�>�p��\��G�@��
                                    Bxkn�  T          @��@��>\��(���(�@��\@��>�׾��
�j�H@���                                    Bxkn��  �          @���@��
>\)��ff�B{?˅@��
>Ǯ�z�H�5@��                                    Bxkn�^  �          @�Q�@�Q�>�녿��R�i�@�(�@�Q�?333�����N�\A                                    Bxkn�  �          @�\)@�  >�׿��H�b�H@�@�  ?B�\��ff�E��A��                                    Bxkn�  �          @�ff@��H?E���  �P  A"{@��H?O\)�L�Ϳ(��A)                                    Bxkn�P  �          @���@�  <#�
�����j=q>\)@�  =�Q쾔z��`��?��                                    Bxko�  �          @�  @�p��\)=�G�?��C���@�p��녽#�
��(�C���                                    Bxko�  �          @�Q�@��R=L��>��
@xQ�?(��@��R��>��
@z�HC��
                                    Bxko%B  �          @�
=@K�?�ff�7
=�{A뙚@K�@���z���ffBG�                                    Bxko3�  �          @�z�@Z�H@Q녿c�
�)�B/p�@Z�H@Y���L�Ϳ��B3z�                                    BxkoB�  �          @���@p  @E>���@�
=B�
@p  @8��?�33A[\)B�                                    BxkoQ4  �          @�Q�@�Q�?��R>�\)@^{A�z�@�Q�?�{?E�A�RA��\                                    Bxko_�  �          @��@�z�?��H<�>��
A�{@�z�?�33>�ff@��A��\                                    Bxkon�  �          @�
=@�Q�?
=�B�\��@�(�@�Q�?B�\�
=��ffA!                                    Bxko}&  �          @�@�z�=�>aG�@3�
?��
@�z�=u>u@Fff?G�                                    Bxko��  �          @�G�@��\?�\�:�H��A��@��\?�녾���J=qA���                                    Bxko�r  �          @��@�
=@{�����Q�A�R@�
=@1G��aG���Bff                                    Bxko�  �          @�\)@�G�@{������
A���@�G�@'���G��aG�A�33                                    Bxko��  T          @��@��@
=��G����HA�\)@��@*=q�^�R�Q�A�p�                                    Bxko�d  �          @�(�@��
@8�ÿ�(��Q��B�@��
@Fff�����BQ�                                    Bxko�
  �          @�p�@���@Tzῦff�_\)B=q@���@b�\��(�����B$�                                    Bxko�  �          @��@vff@fff��  �~�\B,=q@vff@w
=�����\)B4Q�                                    Bxko�V  �          @�ff@q�@o\)��33�B�HB2��@q�@z=q�.{��\B7�R                                    Bxkp �  �          @��
@|��@`�׿Y�����B&p�@|��@g
==�\)?5B)��                                    Bxkp�  �          @�33@���@S�
�s33�%�B�
@���@\(���Q�k�B!Q�                                    BxkpH  �          @��\@u@c�
�5����B+p�@u@hQ�>W
=@�B-�                                    Bxkp,�  �          @�33@}p�@b�\������HB'G�@}p�@c33>�G�@��B'ff                                    Bxkp;�  �          @��
@��@<�Ϳ��JffB	�\@��@I���\���
B�                                    BxkpJ:  
�          @��@���@�Ϳ�33��
=A��
@���@#33���
�4  A�\)                                    BxkpX�  T          @��@��@6ff��33�G�B�
@��@C33�Ǯ��B
=                                    Bxkpg�  �          @�@w
=@n�R��\��p�B0{@w
=@o\)>�@��
B0Q�                                    Bxkpv,  �          @�ff@k�@g���33�x��B2�@k�@e?z�@�\)B1(�                                    Bxkp��  �          @�Q�@X��@hQ���H����B;�R@X��@h��>�ff@�ffB;�H                                    Bxkp�x  �          @�  @g
=@h�ÿc�
���B4�@g
=@p  =��
?W
=B8(�                                    Bxkp�  �          @��@E@�Q�>W
=@z�BX  @E@��?�ffAd(�BR�
                                    Bxkp��  �          @���@333@�ff=L��?
=Bgp�@333@�G�?�Q�AR{Bc�\                                    Bxkp�j  T          @��H@@�\)?�  AZffB�L�@@��@�A�G�B(�                                    Bxkp�  T          @�=q?���@��H?�33A��RB��f?���@�Q�@1�A�p�B�\                                    Bxkpܶ  �          @�?�=q@�\)?ٙ�A�  B�#�?�=q@��
@:�HBz�B�8R                                    Bxkp�\  �          @�(�?��
@�=q@{A£�B�W
?��
@u@Tz�BffB�.                                    Bxkp�  �          @�33@?\)@l��@=qA֣�BLQ�@?\)@<��@Q�B�B2��                                    Bxkq�  T          @�p�?޸R@{�@J�HB�B�=q?޸R@>�R@��\BG{Bn                                    BxkqN  �          @��\?5@���?���A>=qB��3?5@���@'�A�=qB�Q�                                    Bxkq%�  �          @�(�?�ff@��?�=qA��RB�  ?�ff@��@E�B��B�\)                                    Bxkq4�  �          @���@��@���?�A���Bv�H@��@z=q@=p�B�Bg�                                    BxkqC@  �          @�\)@^�R@[�@{A֣�B2��@^�R@*�H@Q�B33B=q                                    BxkqQ�  T          @�p�@�=q@�@��A�{A�p�@�=q?�z�@:�HB��A���                                    Bxkq`�  T          @�  @��
?�ff@0��A�AM��@��
>���@<(�B��@k�                                    Bxkqo2  �          @�ff@�G�?fff@�AՅA4(�@�G�>��@�RA�
=@U�                                    Bxkq}�  T          @��\@\)>�G�?�{A�\)@ȣ�@\)    ?�A�{                                        Bxkq�~  �          @��R@x�ÿ0��?���A��C��)@x�ÿ��\?���Az�\C���                                    Bxkq�$  �          @y��@W
=���?p��A`��C��q@W
=��=q>��@ᙚC���                                    Bxkq��  �          @��@�ff�ff?z�@��
C�ff@�ff��H�u�0��C�
=                                    Bxkq�p  �          @�Q�@����Fff?��A8z�C�N@����P��>B�\@z�C���                                    Bxkq�  �          @�p�@n�R�hQ�?�\)Ak�
C��@n�R�w
=>�Q�@w�C��                                    Bxkqռ  �          @�@l(��p��?��A_\)C�:�@l(��~{>�=q@5C�q�                                    Bxkq�b  T          @�ff@mp��Z=q@   A��HC���@mp��tz�?��\A-C�\                                    Bxkq�  �          @���@P���g�@=qA�ffC��@P�����
?�{AhQ�C�(�                                    Bxkr�  �          @��@HQ���Q�@Q�Aȣ�C�H@HQ����?�p�AJ�\C�o\                                    BxkrT  �          @���@,(����@   A��C�z�@,(����?:�H@�Q�C��                                     Bxkr�  �          @��@1����R@\)A�z�C���@1���z�?xQ�AQ�C��                                    Bxkr-�  �          @�{@{����?�ffA��C��@{���\>�@�z�C��
                                    Bxkr<F  �          @�z�@ ����33?�p�AJ�HC���@ ����  �.{��\C�~�                                    BxkrJ�  �          @��\@z���33?
=@ÅC��@z���=q�L���Q�C��                                    BxkrY�  �          @��@����33=u?!G�C��=@��������YC�                                    Bxkrh8  �          @���@�\����#�
�ǮC��@�\��p���z��m�C�J=                                    Bxkrv�  �          @�  @ff��  �Y����RC�b�@ff���\�
=q��C�J=                                    Bxkr��  T          @�z�@����׿���b�RC�Ǯ@�����#33���HC��                                    Bxkr�*  �          @���?�p�����!G���(�C�Y�?�p��^{�e�'�
C�޸                                    Bxkr��  �          @�?�=q�hQ��`  �"C�ff?�=q�#33��33�X��C��)                                    Bxkr�v  �          @��@���:�H�qG��3�C���@�ÿ����p��_�\C��{                                    Bxkr�  �          @�(�?��ÿ�ff��z�ffC��)?��ýu���\�fC�Ǯ                                    Bxkr��  �          @�  @33��<��
>.{C���@33��  �����a�C���                                    Bxkr�h  �          @���?�  ��33?O\)AQ�C�s3?�  ��(��(���p�C�g�                                    Bxkr�  �          @���?�=q���?��\A+�C��=?�=q�����
=��z�C�h�                                    Bxkr��  �          @���?�=q�N�R@:=qB{C��?�=q�y��?�A�Q�C�f                                    Bxks	Z  �          @���?.{��p�@?\)B{C�e?.{��=q?�A���C��
                                    Bxks   �          @��>�G���p�@\)A¸RC���>�G���33?L��A(�C�XR                                    Bxks&�  �          @�ff?B�\���
?��Ahz�C�=q?B�\��G���G���(�C��                                    Bxks5L  �          @��
?�p����?(��@�C��3?�p���G��B�\���C��R                                    BxksC�  T          @��R?�33���H���R�Q�C���?�33��녿ٙ����C�u�                                    BxksR�  �          @��?��
����@  A�Q�C��f?��
��ff?Y��A��C��=                                    Bxksa>  �          @�G�>��~�R@X��B!  C�P�>���Q�@
=qA\C��f                                    Bxkso�  �          @�Q�?fff���?�p�A�=qC�N?fff��p�?�@�\)C���                                    Bxks~�  �          @�Q�>�������@'�A�z�C���>������\?�Q�AH(�C��q                                    Bxks�0  �          @��>�(��xQ�@g
=B+
=C�.>�(���\)@��A��C��q                                    Bxks��  �          @���@�����
?G�A��C�t{@�����Ϳ����33C�ff                                    Bxks�|  �          @�G�@{���?�R@�
=C�aH@{��z�E���\)C�n                                    Bxks�"  �          @��@{��ff?Q�A	�C��@{��\)�
=��p�C��                                    Bxks��  �          @�
=?������?c�
A33C�Q�?�����\������C�9�                                    Bxks�n  �          @��R@	������?O\)A	G�C��R@	���������=qC�Ǯ                                    Bxks�  �          @��R?�������>�(�@�Q�C��
?������R��=q�6�HC��                                    Bxks�  �          @��?�33���?(��@�(�C��
?�33��G��J=q�	��C���                                    Bxkt`  
�          @���?�Q����׼#�
��G�C�ٚ?�Q��w
=����w\)C�<)                                    Bxkt  �          @�z�@C�
�5�@Q�B�RC�� @C�
�h��@ffA�=qC��                                    Bxkt�  �          @��@�
��{@�\A���C���@�
���\?5@�Q�C���                                    Bxkt.R  �          @���@����z�?�Q�A��
C�Ff@����Q�?�R@���C�S3                                    Bxkt<�  �          @��?Q���
=?&ff@ᙚC�t{?Q���{�Y���\)C�y�                                    BxktK�  �          @���>�ff��G�?333@�C�t{>�ff���׿Tz���HC�w
                                    BxktZD  �          @�p���Q���G�?p��A!G�C����Q����H�������C�\                                    Bxkth�  �          @�p��&ff���?B�\A ��C�� �&ff��녿J=q���C��                                     Bxktw�  �          @��R��{��p�>\@|��C�7
��{������Q��Ip�C�,�                                    Bxkt�6  �          @�Q�������=�Q�?h��C�w
�������ÿ�p��z�HC�p�                                    Bxkt��  �          @�  >L����\)>�33@k�C�>L����33��p��O
=C��                                    Bxkt��  T          @�G�>\����>aG�@G�C���>\��33��\)�ep�C�
=                                    Bxkt�(  �          @��>����p�?@  @��\C�%>�����ͿW
=���C�&f                                    Bxkt��  �          @�ff?#�
��녽�\)�G�C�k�?#�
��녿�������C���                                    Bxkt�t  �          @�{=u��?��A�ffC�O\=u���ͼ��
�L��C�K�                                    Bxkt�  �          @�\)>�{����?�G�AV�\C�ٚ>�{�����
�UC��                                    Bxkt��  �          @�
=����Q�?��
AZ=qC�Ф�����;����J�HC���                                    Bxkt�f  �          @�
=�޸R���\?�G�A���CzB��޸R��z�>�z�@A�C{W
                                    Bxku
  �          @����������R@��A�ffCwuÿ�����z�?:�H@�33Cy0�                                    Bxku�  �          @��׿�G���ff?��HA�{C�ÿ�G�����>�G�@��HC�P�                                    Bxku'X  �          @��׿�p����
@,��A�C���p����R?��RAPz�C�]q                                    Bxku5�  �          @�\)�����{?޸RA���C|������\)>aG�@ffC}��                                    BxkuD�  �          @�\)��\)��\)?�z�A���C~�׿�\)���>�?�Cc�                                    BxkuSJ  �          @�z����z�?�{Ak33C{+������\�\)���HC{��                                    Bxkua�  �          @��Ϳ�ff���\?��
A�  Cy���ff���=#�
>�ffCzs3                                    Bxkup�  �          @�������ff?У�A���C}���������R=�G�?���C~ff                                    Bxku<  �          @�ff�
=���?�G�AW\)C��=�
=��(���{�dz�C��                                     Bxku��  �          @�Q�=�Q���z�?��A-�C�|)=�Q����R�������C�z�                                    Bxku��  �          @�  >�{���R?   @�{C��f>�{�������?�C��                                    Bxku�.  �          @���?.{��p�?O\)A�C���?.{����Tz��
{C���                                    Bxku��  �          @���    ���
?��@��HC�      ��G������6�RC�                                      Bxku�z  �          @�G�>�{��  >�=q@5C��=>�{���H��\)�f�\C�ٚ                                    Bxku�   �          @�G�?(��������
�uC�n?(����\)��z����C��R                                    Bxku��  �          @�Q�?+���ff�8Q���C�|)?+����Ϳ�ff���HC��\                                    Bxku�l  �          @��R?0����(�>aG�@�C���?0����ff��33�n�\C�˅                                    Bxkv  �          @�
=?�����H?�z�Ar�RC�S3?�����þ.{����C��                                    Bxkv�  �          @��?   ��p�?!G�@�=qC��f?   ������
�,��C��                                    Bxkv ^  �          @��
?#�
��Q�?Tz�A��C�J=?#�
��Q�\(��G�C�J=                                    Bxkv/  T          @��
?}p���  >�p�@qG�C��?}p��������W�C�AH                                    Bxkv=�  �          @�=q?�G���Q쿏\)�;�C���?�G����*=q���C���                                    BxkvLP  �          @��\?�ff����5���HC�|)?�ff��{��
�ģ�C��)                                    BxkvZ�  �          @��\?�����Ϳ����H  C�(�?�������1���  C���                                    Bxkvi�  �          @�G�?�\���
�   ���
C���?�\�����]p���C�k�                                    BxkvxB  �          @��?:�H��녿�����C��?:�H����W��Q�C���                                    Bxkv��  �          @�\)?Q������0  ��C��)?Q��e�����>\)C�g�                                    Bxkv��  �          @�?p����z��ff���C�u�?p�������`  �!
=C��)                                    Bxkv�4  �          @��?�=q���H�������C�Ff?�=q�|���`���"�\C��                                     Bxkv��  �          @��\?aG���������ffC�@ ?aG��y���^�R�${C�e                                    Bxkv��  �          @��\>�ff���׿���Q�C��=>�ff�����HQ��G�C��)                                    Bxkv�&  �          @�=q>�����\)��  ��z�C�G�>�����ff�L(���C��3                                    Bxkv��  �          @�녾\)������Q�C�+��\)�~�R�_\)�$�
C��q                                    Bxkv�r  T          @��׾��
����:�H�
=C�녾��
�O\)���
�N��C�5�                                    Bxkv�  �          @�  >u���H�8����C���>u�O\)���H�N\)C�                                      Bxkw
�  �          @���?&ff����5����C�=q?&ff�QG���G��I�RC��                                     Bxkwd  �          @��>�p����H�8���G�C�j=>�p��N�R���H�N33C�=q                                    Bxkw(
  �          @�
=>�z���{�E���C��{>�z��AG���\)�YG�C��{                                    Bxkw6�  �          @�\)?��|(��L(��=qC�AH?��0  ��Q��[�C��                                     BxkwEV  �          @�
=?�=q�����$z���33C���?�=q�P���q��<�C�#�                                    BxkwS�  �          @��?��R��ff�'���\)C�>�?��R�J�H�s�
�@�C���                                    Bxkwb�  �          @�p�@
�H��
=�   ���C���@
�H�XQ��N�R���C�XR                                    BxkwqH  �          @�p�@Q���
=��Q���z�C�e@Q��Y���J�H�{C�
=                                    Bxkw�  �          @�p�?�=q��������C��R?�=q�S33�i���6�C���                                    Bxkw��  �          @�{?��H����xQ��5p�C��?��H��=q�����C���                                    Bxkw�:  �          @�Q�@����H?�33Ax��C�4{@���������G�C���                                    Bxkw��  T          @��R@7��Y��@*�HA�\)C�R@7���=q?���A�=qC��q                                    Bxkw��  �          @�\)@&ff�p��@
=Aڣ�C�O\@&ff��=q?�ffA<z�C���                                    Bxkw�,  �          @��H@�R���
@��A���C���@�R��(�?Tz�A�
C��                                    Bxkw��  �          @���@+��N{@O\)B�C��@+����\@33A�=qC��                                    Bxkw�x  �          @��@5��s�
@(�A�G�C�S3@5���z�?���A=C�o\                                    Bxkw�  �          @�33@*�H�xQ�@!�A�  C�@ @*�H��\)?�z�AH��C�^�                                    Bxkx�  �          @���@E�e�@%A�p�C�n@E��
=?��A`��C�#�                                    Bxkxj  �          @��
@Fff�h��@�AҸRC�9�@Fff��ff?�=qA:�\C�5�                                    Bxkx!  �          @�33@%��^{@G�B(�C�J=@%���G�?�A�p�C��                                     Bxkx/�  �          @��H@(��>�R@l(�B/(�C��=@(���Q�@"�\A�\)C���                                    Bxkx>\  �          @��
?�ff�A�@�
=BP=qC���?�ff��\)@A�B	�C��f                                    BxkxM  �          @�(�?�
=�?\)@��BSG�C�Ǯ?�
=��ff@C33B��C��)                                    Bxkx[�  T          @���@:�H�5�@`��B#(�C��@:�H�s�
@=qA��HC��
                                    BxkxjN  �          @�G�?����\)@�z�By��C��3?���QG�@r�\B9G�C��{                                    Bxkxx�  �          @�33?��Ϳ���@���B�
=C�� ?����S�
@���BD�
C�8R                                    Bxkx��  �          @��
?�{��{@��
B�  C�s3?�{�U@�  BB�\C�7
                                    Bxkx�@  �          @��H?�\�ٙ�@���B�  C�h�?�\�P  @�ffBO�RC��H                                    Bxkx��  �          @�z�?k��˅@��B���C��{?k��HQ�@�ffBP��C�%                                    Bxkx��  �          @�33@  ��@�  B\�\C�  @  �a�@b�\B �C�J=                                    Bxkx�2  �          @���@7��333@�B8�RC��q@7���Q�@A�A��\C�˅                                    Bxkx��  �          @�ff@Dz��Tz�@Y��BC�XR@Dz���Q�@�A�=qC��                                    Bxkx�~  �          @�33?�\)���@��A˙�C��?�\)��{?:�H@�C��                                    Bxkx�$  �          @�G�?����  ?�z�A�ffC��\?�����>u@$z�C��                                    Bxkx��  �          @�p�@=q�w�@9��B�C���@=q���?��HAzffC�˅                                    Bxkyp  T          @���@ff�o\)@N�RB�\C��@ff��33?�A�z�C��f                                    Bxky  �          @�ff@z��mp�@XQ�B(�C��{@z����
?�(�A�C��                                    Bxky(�  �          @��R@&ff�N�R@`��B!  C�ff@&ff���R@\)A���C��{                                    Bxky7b  �          @�G�@5�L��@_\)BC�Ǯ@5��@�RA�=qC�R                                    BxkyF  �          @�Q�@e��L��@,��A�RC�q@e��z=q?�p�Ayp�C�:�                                    BxkyT�  �          @�
=@>�R�J�H@L(�B�HC�� @>�R����?���A�G�C�.                                    BxkycT  �          @�ff@p  �B�\@!G�A܏\C�xR@p  �mp�?�{Ag33C��f                                    Bxkyq�  T          @�{@^�R�Z�H@��A�G�C�@^�R����?�{A=G�C�o\                                    Bxky��  �          @��
@dz��g
=@��AʸRC�` @dz���ff?��A-p�C�0�                                    Bxky�F  T          @�z�?�z���(�?���A�G�C�7
?�z���zὣ�
�W
=C���                                    Bxky��  �          @�p�@N�R�`  ?�p�A���C�U�@N�R�~{?0��@�p�C��3                                    Bxky��  �          @�z�@�G��;�@&ffA�ffC��@�G��hQ�?��HAqG�C�f                                    Bxky�8  �          @�
=@x���(Q�@333A�=qC��\@x���Z�H?޸RA��RC�O\                                    Bxky��  �          @��@mp��X��@p�A��
C��@mp��|(�?n{AQ�C���                                    Bxky؄  �          @�=q@u��9��@1�A�(�C�h�@u��j�H?У�A�Q�C�
                                    Bxky�*  �          @��
@\)�-p�@�Aʏ\C���@\)�U�?�p�AT(�C��                                    Bxky��  �          @���@���!G�?˅A���C�U�@���:=q?#�
@�=qC��H                                    Bxkzv  �          @��@�p��7�?�Q�A��C��H@�p��QG�?(�@љ�C��{                                    Bxkz  
�          @�33@�G��*�H?�{A��C�� @�G��I��?Tz�A	G�C��H                                    Bxkz!�  '          @��@����:=q?���AZ�HC���@����L(�>u@p�C��                                     Bxkz0h  O          @��@�33�G�?�Q�Ap��C�&f@�33�[�>�\)@8��C��                                     Bxkz?  �          @�\)@w
=�n�R?��A2ffC��q@w
=�w���z��@��C�y�                                    BxkzM�  �          @�
=@`  ����?�R@�\)C�
=@`  ����Y���\)C�/\                                    Bxkz\Z  "          @���@x���p  ?��A1G�C��@x���x�þ����HQ�C���                                    Bxkzk   �          @��@?\)���
?G�Ap�C�/\@?\)���
�G��=qC�0�                                    Bxkzy�  �          @�Q�@%����?��
AX��C��@%���������Q�C�'�                                    Bxkz�L  �          @�(�@z����?�z�AiC�Ff@z���������ffC���                                    Bxkz��  �          @��\@E��{?�z�As33C�:�@E��p�����{C��                                     Bxkz��  �          @�z�@G����@ffA��C��{@G����H?
=q@���C��                                    Bxkz�>  �          @�(�@N�R�~�R@33A��C��=@N�R����?E�@��HC��                                     Bxkz��  �          @��@W
=�}p�@\)A�=qC�+�@W
=���?5@�
=C�h�                                    Bxkzъ  �          @�G�@Mp���?�33An=qC���@Mp���������=qC��                                    Bxkz�0  �          @�z�@S�
����@A�G�C���@S�
��  ?��@�(�C�&f                                    Bxkz��  �          @���@�����(�@N{B�C���@����/\)@=qA��
C���                                    Bxkz�|  �          @�Q�@e�\)@\(�B\)C�@e�S�
@�AӮC��                                    Bxk{"  �          @��\@I���r�\@(��A�C���@I����\)?��A;�
C���                                    Bxk{�  �          @���@;����?���A��\C��@;���{>W
=@�C��q                                    Bxk{)n  �          @�  @n{�e�?�{A�{C��@n{��Q�>��@�z�C�l�                                    Bxk{8  �          @�  @a��j�H@�\A���C��=@a����?(�@��
C�&f                                    Bxk{F�  �          @��R@g��dz�?�(�A�G�C��R@g�����?�@���C���                                    Bxk{U`  �          @�
=@s33�S33@�A���C��H@s33�u?O\)A  C�XR                                    Bxk{d  �          @�\)@|���P  ?�Q�A��C�AH@|���n�R?&ff@ڏ\C�P�                                    Bxk{r�  �          @�p�@o\)�a�?�
=A���C�XR@o\)�x��>��R@R�\C���                                    Bxk{�R  �          @�G�@qG��P  ?�(�A�{C���@qG��i��>�G�@�G�C��)                                    Bxk{��  �          @��H@n{�R�\?�A���C�AH@n{�n�R?
=q@�(�C�t{                                    Bxk{��  �          @�(�@tz��[�?˅A�C��@tz��qG�>�  @,��C���                                    Bxk{�D  �          @��@p  �`��?��Aq�C�p�@p  �q�<�>�{C�ff                                    Bxk{��  �          @�(�@`���`��?�(�A��HC���@`���~�R?�@�(�C��{                                    Bxk{ʐ  �          @��\@W
=�[�@�\A�{C�0�@W
=����?fffA�C���                                    Bxk{�6  �          @�{@W����@H��B
=C�J=@W��W�@�
A��C�~�                                    Bxk{��  �          @��@s33�6ff?�Q�A��\C��3@s33�QG�?
=q@�C���                                    Bxk{��  T          @��R@���A�?�  A2ffC���@���K��.{����C��R                                    Bxk|(  �          @��@�=q�1G�?�Q�AZ{C��f@�=q�@��=�G�?���C��H                                    Bxk|�  �          @��@~{�%?��AX(�C�j=@~{�5�=�?�\)C�@                                     Bxk|"t  �          @�Q�@a��s�
?��@��C�j=@a��p  �aG���C���                                    Bxk|1  �          @��@K����
=�G�?��HC���@K�������R��(�C��
                                    Bxk|?�  �          @�Q�@W
=���
�#�
��
=C��=@W
=���������(�C��R                                    Bxk|Nf  �          @���@XQ�������
�Q�C�޸@XQ����ÿ���G�C���                                    Bxk|]  �          @�ff@Mp����;\����C��@Mp��|�Ϳ�(���\)C��3                                    Bxk|k�  �          @�Q�@<���p������ffC��R@<�Ϳ�������(�C���                                    Bxk|zX  �          @x��?���@N{��p����HB�B�?���@hQ��
=�ə�B�\)                                    Bxk|��  �          @��\?�\)@>�R�\)��
BgG�?�\)@fff�z�H�Yp�By�
                                    Bxk|��  �          @�(�@N�R@���+���B�H@N�R@Mp��������B3=q                                    Bxk|�J  �          @�(�@W�@��,(����B33@W�@L(���\)��  B-��                                    Bxk|��  T          @�{@s�
?��R�I���Q�A�\)@s�
@#�
����z�Bz�                                    Bxk|Ö  T          @�
=@��?��R�#�
��(�A���@��@333�У����HB
Q�                                    Bxk|�<  �          @�\)@��?���=p��
�
A��
@��@�  ��
=A�                                      Bxk|��  �          @��R@n{?�(��W
=�"��A�p�@n{@���*=q���HB�                                    Bxk|�  �          @�\)@xQ�?Q��X���"  A=�@xQ�@G��5���A�Q�                                    Bxk|�.  �          @�
=@w
=>����\���&�@�(�@w
=?�z��C33��
A�=q                                    Bxk}�  �          @�Q�@�p��8Q��G����C��@�p�?�\�J=q��
@�p�                                    Bxk}z  �          @�\)@r�\��������C��H@r�\����C�
��C�Ff                                    Bxk}*   �          @�z�@n{��
�&ff���C�q@n{��z��QG�� (�C�]q                                    Bxk}8�  �          @��@`���Dz�޸R���C�h�@`���(��0���(�C��                                    Bxk}Gl  �          @��\@Mp��@�׿�Q����C�k�@Mp��	���,����
C��                                    Bxk}V  �          @�z�@P  �E�������33C�:�@P  �
=q�8Q���C�4{                                    Bxk}d�  �          @�?�=q���׿������C��3?�=q�e�J=q� �RC�Y�                                    Bxk}s^  �          @��H?�z����ÿ�Q���z�C���?�z��c33�Q��"��C���                                    Bxk}�  �          @���?�z�����"�\��  C�y�?�z��5�z=q�K�
C�9�                                    Bxk}��  �          @�Q�@3�
�Q��J=q�"z�C��q@3�
��  �u��PQ�C�:�                                    Bxk}�P  �          @��H@S�
�G��P��� �C�Q�@S�
��R�q��A(�C��\                                    Bxk}��  �          @��H@[��\)�<(���RC�j=@[��s33�e��4�\C�J=                                    Bxk}��  �          @��H@!G��<(��O\)���C�U�@!G���(����
�Z�C�޸                                    Bxk}�B  �          @���@.{�_\)��H��\)C��
@.{��\�dz��4z�C���                                    Bxk}��  �          @���@\)�^�R�*=q����C���@\)�(��s33�C33C�T{                                    Bxk}�  �          @���@3�
�\(�����\)C��)@3�
�33�[��-z�C�Z�                                    Bxk}�4  �          @���@P���5�"�\��C�z�@P�׿���\(��-(�C���                                    Bxk~�  �          @�Q�@XQ��'
=�%��(�C�"�@XQ쿴z��Y���+��C��                                    Bxk~�  �          @�33@J�H�W������Z�RC��q@J�H�*�H���C��                                    Bxk~#&  �          @�(�@xQ��4z����{C���@xQ��#33��(��k33C�W
                                    Bxk~1�  �          @�\)@��
���R�c�
�%p�C�\@��
���
�����\)C��R                                    Bxk~@r  �          @���@�p����Ϳ�p���Q�C��3@�p�>��R��  ��{@r�\                                    Bxk~O  �          @���@�녿��ÿ:�H�
{C�|)@�녿�zῥ��t��C��R                                    Bxk~]�  �          @��R@����#�
�z�H�7�C�0�@����W
=��z��XQ�C���                                    Bxk~ld  �          @��R@�p��������
�z�HC��@�p�����#�
��p�C���                                    Bxk~{
  �          @�{@|�Ϳ�{���H���C�b�@|�Ϳs33�!���G�C�B�                                    Bxk~��  �          @�\)@u�\)�����{C���@u��\���ڏ\C��{                                    Bxk~�V  �          @�\)@c33�7
=�޸R����C��=@c33�����.{��RC��R                                    Bxk~��  �          @�p�@k�����p���=qC���@k���33�0  ���C���                                    Bxk~��  �          @��H@3�
�E�����ٮC�&f@3�
���R�L(��+33C�T{                                    Bxk~�H  �          @��R@]p��J�H��
=���
C�@]p��ff�$z���ffC��\                                    Bxk~��  �          @��@QG��Z=q�������\C��f@QG��#�
�+��ffC���                                    Bxk~�  �          @�z�@{�����{��C�p�@{�4z��hQ��5=qC�q                                    Bxk~�:  �          @�33@���������z�C���@���Fff�g��*��C��3                                    Bxk~��  �          @�=q@ �������p�����C�O\@ ���L���QG��(�C��                                    Bxk�  �          @��
@����\����ٙ�C�|)@����{���H��  C��q                                    Bxk,  �          @��@�
=�z�G��p�C�G�@�
=�������z�C���                                    Bxk*�  �          @���@b�\�XQ�k��+�C�0�@b�\�@  �Ǯ��ffC�ٚ                                    Bxk9x  �          @�\)?������Ϳ#�
��C��q?�������"�\�홚C�                                    BxkH  �          @�\)?   ���\������ffC��3?   �l���fff�0  C�ٚ                                    BxkV�  �          @�  ?O\)��ff����C��?O\)�^{�s33�;\)C��q                                    Bxkej  �          @�\)?J=q���
�.{��C�%?J=q�:=q��  �Zz�C��                                     Bxkt  �          @��@����R�����{C���@��Z�H�S33��\C�\)                                    Bxk��  �          @��@Q���  �	�����RC���@Q��P  �q��0C���                                    Bxk�\  T          @��@.{�z�H�#33��\)C�aH@.{�#33�z�H�9��C�ff                                    Bxk�  �          @�=q@B�\��p���
���
C��@B�\�>�R�dz����C�Ǯ                                    Bxk��  �          @��
@S�
���R��
=���C�3@S�
�J�H�O\)�
=C��                                    Bxk�N  �          @�@[����ÿ����mp�C�Z�@[��Tz��C�
���C��{                                    Bxk��  �          @�@K����\���
�'�C�b�@K��p���333��
=C�
                                    Bxkښ  �          @�z�?������
��z��;�C�&f?�����G������33C�3                                    Bxk�@  �          @�(�?���������\C��
?������33��33C���                                    Bxk��  �          @�33?�����\)>���@��C�@ ?�������\��p�C���                                    Bxk��  �          @��?����Q�?O\)A�
C��f?����(���z��g�C�                                      Bxk�2  �          @�z�?������?���Ac�C��{?�����Ϳh�����C��{                                    Bxk�#�  �          @�(�?k����R?h��A��C�˅?k������z��ip�C���                                    Bxk�2~  �          @�\)?������?�@�{C��f?����Q��ff��G�C��)                                    Bxk�A$  �          @�G�?�Q���ff?h��A��C���?�Q����H��z��aC�˅                                    Bxk�O�  T          @���@i���tz�?�\)A���C�� @i�����ͽ�Q�uC���                                    Bxk�^p  T          @�33@��H�/\)?У�A���C�:�@��H�J�H>�Q�@aG�C�aH                                    Bxk�m  �          @���@����1�?��RA�ffC��f@����W
=?(��@�\)C�u�                                    Bxk�{�  �          @��@�z��C�
?�Q�A��C�O\@�z��e?   @��HC�(�                                    Bxk��b  �          @�=q@����z=q@�\A���C��H@�����Q�>�@��C�ٚ                                    Bxk��  �          @���@�{�c33?��RAjffC��{@�{�u����C��3                                    Bxk���  �          @�=q@�  �U�?�A�Q�C��H@�  �n{>�?�G�C���                                    Bxk��T  �          @���@��j�H?L��@�p�C���@��j�H�L�����C���                                    Bxk���  �          @���@��\�`��?�R@�p�C��)@��\�\�Ϳfff�
=qC�:�                                    Bxk�Ӡ  �          @���@�(��-p�?��HAap�C�
@�(��Dz�>B�\?�C���                                    Bxk��F  �          @��@�{�|(�?�G�A=qC�4{@�{��Q�:�H�߮C��)                                    Bxk���  �          @�G�@��H��=q?^�RA��C��{@��H��G����\�  C�Ф                                    Bxk���  �          @���@&ff���\?z�HA=qC��@&ff��  �����O
=C�.                                    Bxk�8  �          @�33@:�H����?#�
@\C�}q@:�H�����У��z{C��                                    Bxk��  �          @�=q@Q���Q�>�{@Mp�C���@Q���z�� ������C�ff                                    Bxk�+�  �          @�Q�?�p���
=?�(�A;�
C��?�p���ff����JffC��                                    Bxk�:*  �          @�{?У���=q>�{@S�
C�%?У���{�33����C��{                                    Bxk�H�  �          @�  ?333���R�����b�\C��H?333����n�R���C�`                                     Bxk�Wv  �          @��@=q��
=�   ���C��3@=q�i���z�H�'\)C��R                                    Bxk�f  �          @�z�@Z�H����>u@=qC�0�@Z�H��녿�  ���C�9�                                    Bxk�t�  �          @�{@7����H>8Q�?��C���@7���{�   ���C���                                    Bxk��h  �          @��
@Q���=q=�Q�?\(�C�q@Q����Ϳ�(���{C�Y�                                    Bxk��  �          @�\)@Z�H��녿���\)C���@Z�H����#�
����C��q                                    Bxk���  �          @��R@����{��  �(�C��@���W
=�,���أ�C�/\                                    Bxk��Z  �          @���@��
�qG���G��ffC�c�@��
�>�R�!��ř�C���                                    Bxk��   �          @�  @^�R���H�{���C�0�@^�R�0  �p���!{C�ٚ                                    Bxk�̦  �          @�@mp���33����J�RC�=q@mp��XQ��AG����\C��R                                    Bxk��L  �          @���@L(����׿0�����RC�33@L(��U�ff��(�C�ٚ                                    Bxk���  �          @��R@  ��?&ff@˅C�=q@  ��{�޸R��
=C��
                                    Bxk���  �          @��?������?8Q�@��
C��?����������33C�,�                                    Bxk�>  �          @��?��������C���?�����7
=��C���                                    Bxk��  �          @��R?�
=��������33C�"�?�
=�{�����9C�T{                                    Bxk�$�  �          @�{@�R��p����
�H��C��{@�R����XQ��C���                                    Bxk�30  �          @��
@Tz���  ������HC�z�@Tz���
=�(���\)C��                                    Bxk�A�  �          @�z�@���c33?�z�A�  C�
=@���z�H�#�
�\C��H                                    Bxk�P|  �          @��@G
=���\@ffA�z�C���@G
=��p�>u@�C�'�                                    Bxk�_"  �          @��@l���?\)@W
=B�
C���@l������?�(�A��C��q                                    Bxk�m�  �          @��H@������@\(�B(�C�g�@����l��@�A��RC��                                    Bxk�|n  �          @�33@�Q��ff@,��A���C��@�Q��Dz�?�G�An�RC���                                    Bxk��  �          @��@�  �:=q?���A�ffC�:�@�  �Z=q>�Q�@c33C�#�                                    Bxk���  �          @�33@���u?=p�@��C���@���q녿�G��!G�C�q                                    Bxk��`  �          @��H@H����33���
�HQ�C�u�@H����
=�����z�C�P�                                    Bxk��  �          @���@z����R�����(�C�c�@z������;���C�                                    Bxk�Ŭ  �          @�z�@g����
>�\)@0��C��@g����ÿ�\��ffC�"�                                    Bxk��R  �          @�p�@L����p�����ffC�~�@L����
=�$z���\)C��\                                    Bxk���  �          @��R@G
=���H���R�C\)C�]q@G
=�s33�P  ��HC��f                                    Bxk��  �          @�\)@&ff��
=�&ff��ffC�n@&ff�E��p��>=qC��                                    Bxk� D  �          @���?��������Z�H�(�C��{?����#33��=q�m�\C���                                    Bxk��  �          @�@~{��
=�#�
��\)C���@~{�mp�� ����Q�C�xR                                    Bxk��  �          @�z�@��H�N{>�@�  C�1�@��H�E��G��   C���                                    Bxk�,6  �          @�ff@vff�r�\�����C��{@vff�
=�q��C�7
                                    Bxk�:�  �          @��R@�p��A녿�R��z�C�.@�p��(���33���RC��                                    Bxk�I�  �          @�@�  �AG��B�\��C�k�@�  ��������C�Q�                                    Bxk�X(  �          @�z�@aG����\�fff�p�C��@aG��l���5���C��                                    Bxk�f�  �          @�33@vff��=q����  C�ٚ@vff�j�H����\)C�1�                                    Bxk�ut  �          @��@��H�n�R�Y���\)C��=@��H�=p���H��(�C���                                    Bxk��  �          @���@s33��������C�"�@s33�_\)�
=�ĸRC��{                                    Bxk���  �          @��@�Q��xQ�>\)?�\)C��{@�Q��b�\��{���
C�(�                                    Bxk��f  �          @���@����h�ÿ�
=��
=C���@���� ���HQ��(�C���                                    Bxk��  �          @�33@���O\)��H���HC���@�����g���C�R                                    Bxk���  �          @���@�G��n{��z��f�HC���@�G��,(��;�����C�,�                                    Bxk��X  �          @�\)@��H�QG����
�%��C�C�@��H�p�������C���                                    Bxk���  �          @�=q@����ÿ��R�nffC��R@����  ��\����C�5�                                    Bxk��  �          @�33@��ͿxQ쿊=q�*ffC�q@��;\��33�^ffC�\                                    Bxk��J  �          @��\@�
=�!녾\)���C�3@�
=��Ϳ��\�I��C���                                    Bxk��  �          @�=q@��׿Ǯ�Q��{C�!H@��׿��\��
=�e�C��                                    Bxk��  �          @��\@I����{?�33A7�C��{@I����ff�����/�C���                                    Bxk�%<  �          @�=q@a���G�>�=q@.�RC���@a���p�������C��                                    Bxk�3�  �          @�=q@dz���
=?=p�@��C�L�@dz���=q��{�Z�\C��f                                    Bxk�B�  �          @�  @�{�n�R>�=q@1G�C�%@�{�\�Ϳ����o�C�C�                                    Bxk�Q.  �          @���@�Q��n{?�  A�C��\@�Q���������C�#�                                    Bxk�_�  �          @�=q@�G��e?��
A'
=C�@�G��j=q�5��RC�                                    Bxk�nz  �          @�33@�=q�W�?�\)AX��C��=@�=q�hQ쾣�
�FffC��=                                    Bxk�}   �          @���@���B�\?�=qA\)C�N@���Z�H<��
>L��C��3                                    Bxk���  �          @��@_\)��=q?�R@ÅC��\@_\)��33����v�RC�aH                                    Bxk��l  �          @��\@^{��G���\)�3�
C��R@^{�z=q�z�����C���                                    Bxk��  �          @���@����~�R�(�����C�)@����P���
=���HC��\                                    Bxk���  �          @�  @�\)�e>B�\?���C���@�\)�Q녿�p��o
=C��f                                    Bxk��^  �          @���@��\��?�Q�Ag�C��\@��\�.�R>W
=@�C��q                                    Bxk��  �          @�=q@�
=�;�?fffA�RC�� @�
=�@�׿
=q���C�^�                                    Bxk��  �          @���@��H����?���A@(�C���@��H�
�H>k�@��C��                                    Bxk��P  �          @�\)@����33?�Q�Ah��C���@����
=?@  @�  C�\)                                    Bxk� �  �          @�
=@���G�?�z�A�G�C�  @���#�
?
=q@�{C���                                    Bxk��  �          @�ff@��H�/\)?�z�A<Q�C�:�@��H�>{�u���C�=q                                    Bxk�B  �          @�z�@����K�?\(�Ap�C��@����Mp��333��p�C�c�                                    Bxk�,�  �          @��
@�G��C33?c�
Az�C�
=@�G��G���R���C��                                     Bxk�;�  �          @��\@��j�H>8Q�?�C�W
@��U������RC��\                                    Bxk�J4  �          @��@�Q��w�?�@��RC��@�Q��j�H�����Z=qC��                                    Bxk�X�  �          @�=q@����`  >��@�\)C�XR@����R�\��G��R�RC�5�                                    Bxk�g�  �          @���@���\?�33A{\)C�)@�����R?�@�p�C��{                                    Bxk�v&  �          @�Q�@��H��R@"�\A�{C�Y�@��H��p�?��HA�
=C�(�                                    Bxk���  �          @�  @�
=�h��@ffA�(�C���@�
=���
?�
=Ar�\C�&f                                    Bxk��r  �          @�  @��\�33?�AD��C�=q@��\�%��#�
��ffC���                                    Bxk��  �          @���@�{�:�H=u?��C��@�{�'
=����[�
C�s3                                    Bxk���  �          @��@�Q��h�ÿz���C���@�Q��<(��{��33C��\                                    Bxk��d  �          @�Q�@w
=�z�H�\)��p�C�C�@w
=�Y�����H����C�O\                                    Bxk��
  �          @��@u��{�=�?��HC�  @u��aG��޸R���C���                                    Bxk�ܰ  �          @�p�@vff�l��?J=qA
=C��@vff�h�ÿ��
�0��C�N                                    Bxk��V  �          @�  @��
��\@�A�Q�C�� @��
�7�?��A6�RC�\)                                    Bxk���  �          @���@�Q��1�?�Q�As\)C�'�@�Q��HQ�#�
�\C���                                    Bxk��  �          @��R@��H��33@5A�\)C�3@��H�$z�?���A�
=C��\                                    Bxk�H  T          @�
=@��
�{@�A��C�B�@��
�K�?5@�  C���                                    Bxk�%�  �          @�@��\�C�
?��\A-��C�\)@��\�K������=qC�ٚ                                    Bxk�4�  �          @�{@p  �q녿c�
��HC�aH@p  �:�H�$z����HC�\                                    Bxk�C:  �          @�ff@@  ��=q?L��A33C�5�@@  �}p���(��\��C��3                                    Bxk�Q�  �          @�{@+���33?Y��AG�C��q@+���Q쿘Q��]��C��                                    Bxk�`�  �          @�z�@����z�?�(�A���C��@����>�R?!G�@ٙ�C��{                                    Bxk�o,  �          @���@�
=�-p�?L��A�C��@�
=�0�׿z���=qC��
                                    Bxk�}�  �          @��R@�33�6ff����\)C�  @�33�\)������
=C��                                    Bxk��x  �          @�(�@�(��&ff�Q��(�C�T{@�(���33���H����C���                                    Bxk��  �          @��@c33�=p�����(�C��@c33��\)�O\)���C�                                    Bxk���  �          @���@�{�<(�?
=q@��C�}q@�{�5�k��$��C��\                                    Bxk��j  �          @�ff@����*�H>�(�@��\C��@����#33�c�
�"=qC��\                                    Bxk��  �          @�
=@��\���?�G�A�p�C��f@��\�4z�>8Q�@ ��C�q�                                    Bxk�ն  �          @��R@��H�!G�?�ffAhz�C���@��H�5�u�+�C�k�                                    Bxk��\  �          @�
=@�p����?��A;�C���@�p���z�>���@U�C��\                                    Bxk��  �          @��R@�z��33?
=q@�  C��{@�z�ٙ����R�[�C�e                                    Bxk��  �          @���@��Ϳ333?E�A=qC�#�@��Ϳz�H>�{@l��C��q                                    Bxk�N  �          @���@�z�>u?c�
A(�@+�@�z�W
=?c�
Ap�C�Ф                                    Bxk��  �          @��@���=u?У�A��R?&ff@����:�H?���A~ffC���                                    Bxk�-�  �          @�  @��H�#�
?���AHz�C�3@��H�0��?h��A!G�C�&f                                    Bxk�<@  �          @���@�Q�=�?�33Ay��?���@�Q��?��
AdQ�C���                                    Bxk�J�  �          @���@���=�G�?�\)Ar{?�(�@����\)?�  A\��C���                                    Bxk�Y�  �          @�{@�{��G�?5A�C�=q@�{��p�>\)?˅C��                                    Bxk�h2  �          @�33@�\)��    �#�
C��q@�\)�˅�aG����C�(�                                    Bxk�v�  �          @��@�z������\�4z�C��R@�z`\)��z�����C�˅                                    Bxk��~  �          @�=q@���� �׿�ff����C�{@�����G����ԸRC���                                    Bxk��$  �          @��\@u�G��5��HC��
@u�(��e��*�\C�w
                                    Bxk���  �          @��\@�Q��\������RC�G�@�Q��\�.�R��Q�C��H                                    Bxk��p  �          @��@�z�(��}p��,��C��H@�z�L�Ϳ�z��K
=C��{                                    Bxk��  �          @���@�(��(�ÿY���G�C�T{@�(��.{����;\)C��                                    Bxk�μ  �          @���@�p��p�׾��H���HC�ٚ@�p��z�c�
��\C�Ǯ                                    Bxk��b  �          @�=q@���?(��?�33Au@�\)@�������?�ffA�{C�n                                    Bxk��  �          @���@��\?��?��HA�z�AL  @��\=�@�A�{?��H                                    Bxk���  �          @��H@���?���?�  A��\A�  @���?
=q@ ��A�@���                                    Bxk�	T  �          @�33@�{?�\?�=qA���A�G�@�{?&ff@p�A�@��                                    Bxk��  �          @��H@��?�{?��HA��A�{@��?+�@'�A�RA��                                    Bxk�&�  �          @��\@�Q�?��
@��A��A�ff@�Q�>�@2�\A���@��H                                    Bxk�5F  �          @�\)@�33>��8Q��  @��R@�33?�\=��
?s33@ə�                                    Bxk�C�  �          @��R@�z�>L�Ϳ�Q���ff@ff@�z�?�  �����v�\A9�                                    Bxk�R�  �          @�ff@��׿   ����C��@���>\�ٙ�����@��H                                    Bxk�a8  �          @��@�p��+���ff��G�C��@�p�>�����33���R@j�H                                    Bxk�o�  �          @��@��\>�
=>��@��@���@��\>.{?\)@��?�                                    Bxk�~�  �          @�p�@�G�?(��?fffA!p�@�p�@�G�>�?�{AF�R?���                                    Bxk��*  �          @��@�>�ff?�Q�AX��@���@��B�\?�G�AeC��                                    Bxk���  �          @�p�@�ff�J=q?k�A&�RC�p�@�ff��33>��@�(�C�}q                                    Bxk��v  T          @��R@�����Ϳ����Z�HC�#�@�����������Q�C��                                    Bxk��  �          @���@�(����>��@�Q�C���@�(���z�8Q����C��f                                    Bxk���  �          @���@��R���?!G�@�G�C��)@��R������Q쿅�C�ٚ                                    Bxk��h  �          @���@��>���?޸RA��
@w
=@����?�
=A��C��\                                    Bxk��  �          @��
@�ff?��H@33A���Ag�@�ff���
@&ffA癚C��H                                    Bxk��  �          @�33@�{?�33@��A�ffA[\)@�{��@!�A�z�C�C�                                    Bxk�Z  �          @�(�@��\?W
=@�A�p�A��@��\��33@33A˅C��                                    Bxk�   �          @�Q�@�Q�?aG�@!�A���A0��@�Q���H@(��A�G�C��                                    Bxk��  �          @���@�z�>B�\@�RA��H@
=@�zῇ�@  A�Q�C���                                    Bxk�.L  �          @��R@�z�>�{?ٙ�A��@~{@�z���?�33A�  C���                                    Bxk�<�  �          @�\)@�����
@Q�AڸRC���@����  @�A�Q�C�h�                                    Bxk�K�  �          @��@���L��@(�A�\)C��f@�����@G�A���C���                                    Bxk�Z>  �          @���@�33��\)?���A���C�L�@�33�Q�?c�
A  C�W
                                    Bxk�h�  �          @��H@�{��?�=qAf�\C��@�{�#33=�\)?@  C��q                                    Bxk�w�  T          @�=q@����$z�?���A��C�  @����I��>��
@b�\C�k�                                    Bxk��0  �          @��@�{�,(�?�z�AI��C�ff@�{�:=q�����fffC�b�                                    Bxk���  �          @�(�@R�\�vff?��HAW�C�@ @R�\�|(��^�R���C��=                                    Bxk��|  T          @��
@<�Ϳ�\)@33B=qC���@<����?�
=A�=qC�,�                                    Bxk��"  �          @�
=@2�\?��@��HBV��A�p�@2�\�Tz�@���Be{C��3                                    Bxk���  T          @�p�@Dz�?�\)@���BD��A���@Dz�(�@��BW��C�c�                                    Bxk��n  �          @��@Vff?�=q@��HBD�HA�G�@Vff��{@��\BD�C��H                                    Bxk��  �          @�Q�@L(�?���@�Q�BK
=A��R@L(��p��@��
BS  C��                                    Bxk��  �          @�G�@_\)=�\)@�Q�BJ��?���@_\)�z�@n{B*=qC���                                    Bxk��`  �          @�p�@\(���ff@�z�BH
=C�C�@\(��{@W
=B�\C�0�                                    Bxk�
  �          @�(�@dz�?0��@z�HB<�A/�
@dz`\)@o\)B1ffC���                                    Bxk��  �          @�  @}p���@W�B!Q�C�
=@}p���@6ffB��C��H                                    Bxk�'R  �          @��R@����\)@]p�B�RC�{@����p�@6ffA�  C�Ff                                    Bxk�5�  "          @�@��Ϳ���@UB\)C��)@����(Q�@�AÙ�C��3                                    Bxk�D�  �          @���@i���`  @�AŅC��@i����ff>�  @!�C��                                     Bxk�SD  �          @�33@�G��XQ�?ٙ�A��C��@�G��q녾u�(�C�p�                                    Bxk�a�  �          @�=q@��
�Mp�?s33A�C�� @��
�P  �J=q�Q�C���                                    Bxk�p�  �          @�{@��H�*=q?�(�A�C��3@��H�J=q>\)?�(�C��{                                    Bxk�6  �          @��H@�ff�J=q@G�A���C�N@�ff�p  >��?�p�C��                                    Bxk���  �          @�ff@vff�e�@Q�A��RC��=@vff���<�>�{C�]q                                    Bxk���  �          @�
=@�G��W
=?Q�A��C��H@�G��S�
�}p���C��                                    Bxk��(  �          @�z�@���Tz�>�@�\)C��H@���Fff���
�S�C���                                    Bxk���  �          @��@����W
=?   @��\C���@����I�����\�O\)C���                                    Bxk��t  �          @�z�@����R�\>��
@P  C�3@����@  ����e�C�Q�                                    Bxk��  �          @�(�@��\�P  >#�
?�z�C�P�@��\�8Q��G��x��C��                                    Bxk���  �          @��
@���q녾���)��C���@���HQ�������C�33                                    Bxk��f  �          @��H@�z��AG�@33A�z�C���@�z��qG�>�ff@�z�C��                                    Bxk�  �          @�Q�@�p��5@�
A�C���@�p��g�?��@��RC�Y�                                    Bxk��  �          @�ff@u�i��?�A���C�=q@u��녾����A�C��{                                    Bxk� X  T          @�33@%���>�=q@/\)C��f@%����G����RC�XR                                    Bxk�.�  �          @�p�@@�����;8Q��C�u�@@���z=q�!G���Q�C��=                                    Bxk�=�  �          @��@P�����\(���HC�,�@P���XQ��@  �ffC��                                    Bxk�LJ  �          @���@XQ���\)�aG��(�C�L�@XQ��L(��:=q� p�C�P�                                    Bxk�Z�  �          @���@{��q녿�  �%�C��@{��/\)�1���p�C���                                    Bxk�i�  �          @�{@p���w
=��G����
C�'�@p���E���\)C�J=                                    Bxk�x<  �          @�{@����\�ͿW
=��
C���@����!���R��=qC��                                    Bxk���  �          @�@n�R�xQ��R�ӅC��@n�R�AG��!G���C��                                     Bxk���  �          @��R@����fff�������C�  @����2�\�ff����C��=                                    Bxk��.  �          @���@�ff�e�����\C��=@�ff�@  ���H��  C�<)                                    Bxk���  �          @�=q@�ff�g
=>8Q�?�\)C��@�ff�L(���Q���{C�e                                    Bxk��z  �          @��H@�  �r�\>��H@��C�E@�  �`�׿\��C�b�                                    Bxk��   �          @�  @8����33�aG��G�C�{@8���u�"�\��
=C�~�                                    Bxk���  �          @���?��R��  ��p��N�RC��R?��R�j=q�g��${C�C�                                    Bxk��l  �          @��\?��R��Q�Tz��33C�ٚ?��R�xQ��Q����C���                                    Bxk��  �          @��
?������{����C�  ?���,(���\)�W(�C��                                    Bxk�
�  �          @��R?���c�
�|���;�C�'�?�����
��
=��C��{                                    Bxk�^  �          @���?�\)��Q��33��p�C�f?�\)�9����p��X(�C���                                    Bxk�(  �          @��
?��R���H��Q��D(�C��=?��R�p  �hQ��"=qC��3                                    Bxk�6�  �          @�G�?�������Q��FffC�q�?����u��k��&C�*=                                    Bxk�EP  �          @���?�\)������
�,Q�C��{?�\)�x���b�\�   C�H�                                    Bxk�S�  �          @�z�?������ÿ#�
��{C��=?����~�R�H����C�                                    Bxk�b�  �          @���?�  ��녿��H�X(�C��f?�  �`  �aG��*��C��
                                    Bxk�qB  �          @�@33��G�������C�l�@33�g��1���C�0�                                    Bxk��  �          @���@.{���
�W
=�  C�8R@.{�vff�$z���z�C���                                    Bxk���  �          @�{@?\)��=q�u���C��f@?\)��Q��,(���\C�]q                                    Bxk��4  �          @���@6ff���?��@�G�C��q@6ff���ÿ�
=����C��)                                    Bxk���  �          @�G�@>{��33?(�@�z�C�k�@>{��Q������33C�t{                                    Bxk���  �          @��R@����>��
@X��C�Ff@����\�\)��=qC���                                    Bxk��&  �          @�p�@	����{>���@^{C��H@	���������ǅC�f                                    Bxk���  �          @��?������(���z�C�k�?����  �:�H��
C��{                                    Bxk��r  �          @�@/\)��G�?�  A*�\C��3@/\)���Ϳ�  ���HC��R                                    Bxk��  �          @��R@AG���z�?O\)A
=qC�@ @AG���p���=q����C��                                    Bxk��  �          @���@r�\�s�
?�ADQ�C�l�@r�\�w��z�H�#�C�9�                                    Bxk�d  �          @��@G����?�33A?�C��=@G��������Z�HC���                                    Bxk�!
  �          @�G�@���Q�u�#�
C�~�@���  �)����p�C�e                                    Bxk�/�  �          @��
@'���=q?�p�A��\C���@'����R�z�H�+33C�:�                                    Bxk�>V  �          @��@
=���@�G�Bp�C��H@
=�R�\@Tz�BQ�C��{                                    Bxk�L�  �          @�\)@J=q�B�\@N�RB�\C�f@J=q����?�
=AH(�C�5�                                    Bxk�[�  �          @���@'��fff@K�B{C��@'���\)?W
=AC�}q                                    Bxk�jH  �          @���@��}p�@C33B��C�p�@���
=?
=q@�\)C�޸                                    Bxk�x�  �          @��@2�\�|��@�RAمC���@2�\���=u?.{C�l�                                    Bxk���  �          @�33@����@0  A��HC��R@���{>W
=@{C��f                                    Bxk��:  �          @��H@������?�G�AT��C�G�@�������z��n=qC�aH                                    Bxk���  �          @��
@=q��  ?B�\@�=qC��H@=q��p����H��\)C���                                    Bxk���  �          @��@(���=q>�  @#33C���@(����R�(���
=C��                                    Bxk��,  �          @�(�@�����;����U�C��=@����\)�>{��G�C���                                    Bxk���  �          @��\@���G��u��C�h�@���p��5�����C��{                                    Bxk��x  �          @���@G
=��
=?h��A�
C�j=@G
=��G��Ǯ��
=C���                                    Bxk��  �          @�@Fff���\?\(�A�HC���@Fff��z���
���C�e                                    Bxk���  �          @�  @Z=q����>�p�@xQ�C�Ff@Z=q�u�����C�˅                                    Bxk�j  T          @�z�@@  ��>�=q@:=qC��@@  �z�H���
=C���                                    Bxk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxk�(�   �          @��@�z��Q�>���@��C��@�z��@�׿�\)�pQ�C�H                                    Bxk�7\  �          @�\)@  ���?�\A�Q�C��@  ��33�=p��Q�C��                                    Bxk�F  �          @�=q@7���z�?8Q�@��C��{@7����
��Q����\C�n                                    Bxk�T�  �          @�ff@0  ��=q?z�@�
=C�B�@0  �~�R��\��ffC�T{                                    Bxk�cN  �          @��@(�����>k�@%C��@(��~�R����G�C�ff                                    Bxk�q�  �          @�{?�\)���H?�z�APz�C���?�\)��\)�����\)C�R                                    Bxk���  �          @�{@'
=��=q>��@�z�C���@'
=�x�ÿ����
C��                                    Bxk��@  �          @��@O\)��Q�?J=qA
�HC�w
@O\)�tzῷ
=�\)C�                                      Bxk���  �          @���@w
=�S33>��@<(�C��
@w
=�<�Ϳ�G���=qC�N                                    Bxk���  T          @��\@��\�U�>��?�{C�e@��\�:=q��\)��p�C�@                                     Bxk��2  �          @���@Z�H�s�
?8Q�@��C��)@Z�H�g�����y�C��
                                    Bxk���  �          @��?�  ��Q�@ ��A�Q�C���?�  ��z�(����Q�C��3                                    Bxk��~  �          @���@�R���H?��HA�p�C��{@�R��\)�(����C��q                                    Bxk��$  �          @�Q�@XQ��7
=@(Q�A�C�ٚ@XQ��tz�?:�HA�HC��q                                    Bxk���  �          @��\@`  �Mp�?�Q�A�z�C��R@`  �g���  �8��C��                                    Bxk�p  �          @�\)@���p�@�A��
C���@���R?���AT��C�^�                                    Bxk�  �          @��@��R��G�?�(�A���C��@��R�(�?+�@�\C���                                    Bxk�!�  �          @���@tz���@\)A�Q�C��)@tz��P  ?k�A&�HC�Ф                                    Bxk�0b  �          @�Q�@s�
�*�H?J=qAp�C�u�@s�
�,(��=p��Q�C�e                                    Bxk�?  �          @�Q�@1녿��H@S33B=�C�1�@1��1G�@
=qA��C��=                                    Bxk�M�  �          @��@B�\>�p�@`��BC�@�(�@B�\���@K�B,
=C���                                    Bxk�\T  �          @��
@Fff@�@  A�z�B�@Fff?
=q@?\)B.�A=q                                    Bxk�j�  �          @�
=@Q�@.{?�33A�33B�R@Q�?�Q�@5Bp�A�                                      Bxk�y�  �          @��@z�H?��H@   A�=qA�(�@z�H>�@AG�Bff?�Q�                                    Bxk��F  �          @�\)@s33@��@�RA�(�A��
@s33?=p�@E�B=qA0��                                    Bxk���  �          @��@N�R��@P  B-�C��@N�R�-p�@��A�p�C��)                                    Bxk���  �          @��R@"�\>��
@qG�B_p�@���@"�\��(�@W�B>�HC��                                    Bxk��8  �          @�ff?ٙ�@%�@W�B=�Bbz�?ٙ�>��@�\)B�AY                                    Bxk���  �          @��?��H@e�@FffB33B�Ǯ?��H?�33@���B��B�R                                    Bxk�ф  �          @�ff?���@,(�@k�BI=qBv��?���>�{@���B���AS�                                    Bxk��*  �          @��R@#�
@^�R?�\A�(�BV\)@#�
@G�@UB6��B�R                                    Bxk���  �          @���@:=q@C33?��A�(�B9ff@:=q?�(�@@��B&�A���                                    Bxk��v  �          @�(�@"�\?z�H@UBKG�A�33@"�\�n{@W
=BLz�C��                                    Bxk�  �          @��R@>{@7
=@.{B�RB/��@>{?}p�@u�BJ�HA�                                    Bxk��  �          @�=q@Mp�@333@*=qB �B$p�@Mp�?xQ�@o\)B@��A�=q                                    Bxk�)h  �          @��@@n�R@�HA��BgQ�@?��@���BU{B��                                    Bxk�8  �          @��R@�R@`  @)��BQ�Bf
=@�R?���@��Ba��Bff                                    Bxk�F�  
�          @�=q?Y��@�
=@�AΏ\B�\)?Y��@�R@�  Bc33B�W
                                    Bxk�UZ  �          @�33?�z�@E@B�\B��Bh�R?�z�?�G�@��RB{�Aޏ\                                    Bxk�d   �          @���@p�?�33@�G�BX�B"G�@p���@�B{(�C�Ff                                    Bxk�r�  �          @�z�?��@$z�@�\)B^G�By�?���u@�ffB��{C��3                                    Bxk��L  �          @���?k�@2�\@�B[�B���?k�>��@���B�  A��                                    Bxk���  �          @�Q�?�=q@(�@i��BO��BHp�?�=q���
@�Q�B�ffC���                                    Bxk���  �          @��
@e�<�@L��B'(�?
=q@e���@.�RB
�C���                                    Bxk��>  �          @��@�>W
=@�A�G�@4z�@��s33?�z�A�{C��
                                    Bxk���  �          @��
@�
=>aG�?��A�{@3�
@�
=�!G�?�A�  C��)                                    Bxk�ʊ  �          @�  @z=q?˅@'
=A���A��R@z=q�#�
@C33B�
C���                                    Bxk��0  �          @�=q@z�H@z�@�Ȁ\A�\)@z�H?O\)@E�BQ�A;�
                                    Bxk���  �          @��\@QG�@N{?�\A�Q�B233@QG�?��@Mp�B"�A�                                    Bxk��|  �          @��
@1�@Z=q@$z�A�{BK=q@1�?��
@~�RBM�HA�ff                                    Bxk�"  �          @��H@��@,��@U�B)��B>��@��?�@�  Bm��A?�
                                    Bxk��  �          @�Q�?�G��&ff@�
=B���C�l�?�G��:�H@\��B9Q�C���                                    Bxk�"n  �          @��?Ǯ��p�@�G�Bnp�C�  ?Ǯ�|��@%A��RC��H                                    Bxk�1  �          @�z�@z��z�@w
=BSG�C�y�@z��s33@
�HA���C�C�                                    Bxk�?�  �          @�
=?����{@�  BXp�C��q?�����Q�@��A�z�C�b�                                    Bxk�N`  �          @�
=@2�\����@qG�BI��C��H@2�\�J�H@p�A�  C��3                                    Bxk�]  �          @�=q@E��L��@\)BQz�C���@E��
=q@W
=B'{C�xR                                    Bxk�k�  �          @�\)@���G�@�ffB�8RC�� @��%@z�HBDC��                                    Bxk�zR  �          @��H?�(���\)@�{B�p�C��?�(��,(�@���BL��C�"�                                    Bxk���  �          @�33?�{����@�\)B�B�C���?�{�>�R@�Q�BC�C��
                                    Bxk���  T          @�ff?�G�>�@�=qB��=Ah��?�G���
@�G�Be��C���                                    Bxk��D  �          @��\?��R?���@�p�B���B&\)?��R��  @��
B�#�C�j=                                    Bxk���  �          @��\��\)@�@��HB��B�k���\)�:�H@�Q�B�.C�q�                                    Bxk�Ð  �          @�G�>�@��@���B�ǮB�\)>��0��@�\)B��C�5�                                    Bxk��6  �          @�=q>�@\)@���B���B��{>��#�
@��B�� C�s3                                    Bxk���  �          @��H>�33?�@���B��B�W
>�33��p�@���B�W
C���                                    Bxk��  �          @�33?�@�\@��HB��B��
?��Y��@�{B�33C��                                    Bxk��(  �          @��\�#�
@l(�@uB8z�B�aH�#�
?�=q@��RB�z�B�L�                                    Bxk��  �          @��\��@E@�=qBYB�\)��>��
@���B��RBܸR                                    Bxk�t  �          @��\�u@)��@��B^p�B��f�u=L��@�z�B��qC0�=                                    Bxk�*  �          @��R���R@P��@n�RB8p�B�#׿��R?G�@�z�B�
=Ch�                                    Bxk�8�  �          @���c�
@C�
@�33BPffB�k��c�
>��@�33B�33C�                                    Bxk�Gf  �          @���<��
@G�@��B���B���<��
�^�R@�{B��C���                                    Bxk�V  �          @�=q?=p�@J�H@r�\BE�B�?=p�?(��@���B�\B(                                      Bxk�d�  T          @�(�?E�@8��@��
BX
=B�8R?E�>��@���B��A��                                    Bxk�sX  �          @�p�?���@��@1G�BG�B���?���@ ��@���Bw  B^33                                    Bxk���  �          @�p�?�@fff@S�
B"�B���?�?��@��RB�ǮB(�H                                    Bxk���  �          @�{?�ff@.�R@z�HBH�Bbz�?�ff>aG�@���B�u�@��                                    Bxk��J  �          @���?˅@}p�@4z�B�B�W
?˅?���@��Bv�BE��                                    Bxk���  �          @�p�@Q�@\��@J�HB
=Bi�@Q�?�p�@���Bu��A�z�                                    Bxk���  �          @�?�ff@���@#�
A��HB�33?�ff@�
@�=qBlBT��                                    Bxk��<  �          @��?�z�@l(�@?\)B(�Bzff?�z�?��
@�  BuB��                                    Bxk���  �          @�{?�33@���@8��B  B���?�33?��@�33B{��BU��                                    Bxk��  �          @�  ?�33@c33@c33B+��B���?�33?�\)@�z�B�B�                                    Bxk��.  �          @�  ?�
=@E@xQ�B?ffBu?�
=?��@�B��\A�ff                                    Bxk��  �          @�@G�@[�@L(�B�Bn(�@G�?���@���Bzp�A�z�                                    Bxk�z  
�          @�(�@�@@��@W�B&
=BP��@�?B�\@��RBt�RA��                                    Bxk�#   �          @��?�p�@5�@k�B;�B\G�?�p�>�G�@��
B���AIG�                                    Bxk�1�  "          @�=q>�{@l(�@L(�B"�
B��>�{?�
=@�p�B�{B�Q�                                    Bxk�@l  �          @��\�.{@^�R@g
=B8  B����.{?��
@���B�
=B�                                      Bxk�O  T          @�=q?^�R@��@'
=A�ffB�Q�?^�R@��@�By�B��                                    Bxk�]�  T          @���?���@�G�@0  B��B�z�?���?��H@�\)B}=qBr��                                    Bxk�l^  �          @��?��H@o\)@O\)B�B�G�?��H?�Q�@��B��RBHp�                                    Bxk�{  
�          @�{?��@\)@7�B1�Bh�
?��?z�@p��B��A�=q                                    Bxk���  "          @�G�@J=q�Q�@3�
B�C��@J=q�R�\?�  Atz�C��                                    Bxk��P  �          @�{@N{���@R�\B'
=C�u�@N{�QG�?�=qA�=qC�K�                                    Bxk���  �          @���@R�\���H@W�B)=qC�:�@R�\�P  ?�Q�A��RC��                                    Bxk���  
�          @��H@@���=q@R�\B!�
C��@@���r�\?�  A�(�C�>�                                    Bxk��B  �          @���@,���0��@J�HB��C�&f@,������?�(�Aap�C��                                    Bxk���  "          @�{@z=q�(�@8Q�Bp�C��=@z=q�33@�A�
=C�"�                                    Bxk��  T          @��@y���&ff@>{B�RC�J=@y���Q�@
�HA�ffC���                                    Bxk��4  
�          @�(�@c�
�B�\@Q�B*�\C�o\@c�
��z�@*�HB�C���                                    Bxk���  
Z          @���@�Q�>���@"�\B ��@�{@�Q쿁G�@
=A��
C��                                    Bxk��  �          @��@�z�?�ff?�\)A�
=A�(�@�z�?L��@�A�\)A.�\                                    Bxk�&  
�          @�
=@�=q?�(�@�A�  A��@�=q>�33@+�B�@�p�                                    Bxk�*�  	�          @��
@�\)���
@&ffA��HC�*=@�\)�  ?�z�A��\C���                                    Bxk�9r  T          @�G�@Q녿�  @Dz�B33C��\@Q��G�?�A�  C�9�                                    Bxk�H  
�          @�
=@G���@J=qB��C���@G��a�?\A�G�C���                                    Bxk�V�  "          @��@L�����@5�B  C���@L���a�?�\)AT  C�q                                    Bxk�ed  
�          @�
=@ ����
=>Ǯ@�G�C�c�@ ���q녿�z�����C���                                    Bxk�t
  �          @��H?�\��@hQ�BR��C���?�\�n{?���A�=qC��)                                    Bxk���  �          @��R?���8��@Z=qB2�C���?����Q�?���A�C���                                    Bxk��V  �          @��H?������@��B~�\C��?���o\)@<��BffC�1�                                    Bxk���  �          @��H?�G�� ��@���BhffC�� ?�G��\)@$z�A�z�C��                                    Bxk���  
�          @�(�?��=p�@Q�B+��C�� ?���Q�?��HAhQ�C�*=                                    Bxk��H  �          @��\@   �c�
@
=A�(�C���@   ����=�G�?���C��f                                    Bxk���  �          @�{@AG��P  @A�p�C�l�@AG��w
==#�
>�C���                                    Bxk�ڔ  �          @�z�@-p��2�\@0��BffC�
@-p��u�?\(�A%C��f                                    Bxk��:  �          @�33@%�333@>{B��C�^�@%�|��?��
AD��C���                                    Bxk���  �          @�{@:�H��@QG�B'��C���@:�H�e�?�{A�  C��
                                    Bxk��  �          @�ff@!G�����@q�BOQ�C�R@!G��U�@�A��C��3                                    Bxk�,  
�          @�
=?��H�Tz�?��A���C��)?��H�XQ�\(��QC���                                    Bxk�#�  
�          @i�����7��\���
C�����У��3�
�j��C|B�                                    Bxk�2x  	�          @r�\��(������*=q�9��Cc33��(��#�
�O\)�w��C9h�                                    Bxk�A  "          @\)��;Ǯ�O\)�]�\C=����?���<���BG�C��                                    Bxk�O�  �          @o\)���ÿ�  �;��[��Cn������=u�Z=q8RC0�
                                    Bxk�^j  
1          @Vff?���������33C���?����  �=q�S  C�|)                                    Bxk�m  �          @p��@��*�H�������C���@����
�%�1  C�u�                                   Bxk�{�  
�          @��@8Q��1G��������C�@8Q�У��'��Q�C�:�                                    Bxk��\  
c          @��@@  �,(�����33C�3@@  ��z��=q�z�C��H                                    Bxk��  
�          @�(�@L(��  ?�@��
C�q�@L(����@  �.�\C���                                    Bxk���  
�          @��@��H�c�
��G�����C�ٚ@��H=#�
��  ����?
=                                    Bxk��N  T          @��R@��
�Tzὣ�
�xQ�C���@��
�(�ÿ   ��\)C���                                    Bxk���  
1          @��H@��R��  ?#�
A��C�9�@��R��{��  �L(�C���                                    Bxk�Ӛ  
�          @�33@����{?��A��
C��H@����>�@��RC�k�                                    