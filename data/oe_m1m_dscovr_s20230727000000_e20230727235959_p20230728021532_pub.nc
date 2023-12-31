CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230727000000_e20230727235959_p20230728021532_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-28T02:15:32.838Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-27T00:00:00.000Z   time_coverage_end         2023-07-27T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�JR�  b          @�����
=��׾��~�RCGL���
=��\���R�%�CG�                                    Bx�Jaf  
�          @�
=�љ�����
=�^{CG���љ��
=��  �33CG��                                    Bx�Jp  �          @�{��
=�   ���ͿO\)CI(���
=�   =�Q�?8Q�CI(�                                    Bx�J~�  "          @�ff����33=u>��CGW
�����\>k�?�\)CGB�                                    Bx�J�X  �          @�{���
��
�L�;��CEO\���
��
=���?\(�CEJ=                                    Bx�J��  "          @�(��ҏ\��(�>\)?�Q�CD��ҏ\���H>�z�@=qCD��                                    Bx�J��  
�          @�(���z��  >�?�{CB��z�޸R>��@(�CB��                                    Bx�J�J  �          @ۅ��p���(�?!G�@�Q�C@ff��p���z�?=p�@�p�C?��                                    Bx�J��  �          @�
=��Q쿪=q?�G�A	�C?����Q쿠  ?���A33C>�
                                    Bx�J֖  �          @�(�������?�G�A  C?Ǯ�����G�?�{AC?�                                    Bx�J�<  T          @�z���p���
=?n{A z�C@�=��p�����?��
A33C?��                                    Bx�J��  �          @�(����Ϳ�?xQ�A�HC@�����Ϳ���?�=qA��C?�)                                    Bx�K�  
�          @�����
=?��A�HC@��������?���A%��C?�{                                    Bx�K.  "          @�(���(�����?��A�HC@��(���{?�Q�A&{C@�                                    Bx�K�  �          @Ӆ�ə����
?��\A1��CA�)�ə���?��AA��C@��                                    Bx�K.z  �          @����ȣ׿��
?�ffAX��CA���ȣ׿�33?�Ah��C@��                                    Bx�K=   �          @�
=��{���
?��AG�CAY���{��Q�?���A%�C@�)                                    Bx�KK�  �          @��H���
����?+�@�33CA^����
�\?J=q@��
C@�f                                    Bx�KZl  �          @�{�ָR��Q�>��H@���CB(��ָR��33?�R@��
CA�\                                    Bx�Ki  �          @�{��\)���>�Q�@@  C@����\)��G�>�@\)C@��                                    Bx�Kw�  T          @���Q쿣�
?B�\@�=qC>����Q쿜(�?\(�@�(�C>5�                                    Bx�K�^  T          @��
��{����?L��@���C?p���{���?fff@��C>��                                    Bx�K�  �          @��
��\)��
=?5@�p�C=�3��\)��\)?L��@�C=s3                                    Bx�K��  �          @�z��׮��
=?=p�@�{C=��׮��\)?Tz�@�ffC=ff                                    Bx�K�P  �          @����\)����?G�@�  C?O\��\)���
?c�
@��
C>�                                    Bx�K��  �          @�(���\)����?��@�(�C?)��\)���\?(��@��C>��                                    Bx�KϜ  �          @�{��G����
?�@���C>����G����R?!G�@��C>T{                                    Bx�K�B  �          @�  �ڏ\���?\(�@��C>���ڏ\���H?u@�z�C>�                                    Bx�K��  �          @�����(���33?Y��@�ffC=����(���=q?p��@�ffC<�                                    Bx�K��  �          @�Q���(���Q�?#�
@�
=C=�=��(�����?:�H@��C=W
                                    Bx�L
4  �          @޸R���
�s33���
�+�C;�����
�z�H��  ��C<{                                    Bx�L�  �          @�\)��p��Y����G��^�RC:�R��p��Y���#�
����C;                                    Bx�L'�  �          @�  ��{�\(���G��aG�C;���{�^�R�#�
����C;�                                    Bx�L6&  �          @޸R���Ϳc�
���ͿQ�C;aH���Ϳfff���W
=C;k�                                    Bx�LD�  �          @��ۅ�(��333����C9)�ۅ�+��#�
���\C9��                                    Bx�LSr  �          @�p����
���@  ��\)C7޸���
���5��(�C8^�                                    Bx�Lb  �          @�ff����B�\��z��Q�C:=q����G��k���{C:k�                                    Bx�Lp�  �          @�
=��p��O\)�aG���C:�R��p��Tz������\C:ٚ                                    Bx�Ld  �          @޸R���ͿaG�<�>��C;=q���Ϳ^�R=���?\(�C;0�                                    Bx�L�
  �          @�\)���J=q���ͿL��C:����L�ͼ���  C:�\                                    Bx�L��  �          @޸R����8Q쾮{�5�C9�����=p���\)�z�C:#�                                    Bx�L�V  �          @�ff���Ϳ(�þ�(��c�
C9s3���Ϳ0�׾�p��E�C9��                                    Bx�L��  �          @�
=���B�\���R�!�C:E���G��u���RC:u�                                    Bx�LȢ  T          @�  ��ff�=p�������C:���ff�B�\�k���33C:@                                     Bx�L�H  �          @�{��(��c�
��G��s33C;Y���(��c�
�#�
���RC;ff                                    Bx�L��  �          @�ff��(��xQ�<��
>��C<  ��(��u=���?W
=C;��                                    Bx�L��  T          @�{���
��  <#�
=�G�C<O\���
��  =���?Tz�C<E                                    Bx�M:  �          @�{���
�s33<#�
=uC;�H���
�s33=�Q�?=p�C;ٚ                                    Bx�M�  T          @�����H�p�׼#�
����C;�{���H�p��=�\)?z�C;�\                                    Bx�M �  �          @���ۅ�L�ͽL�;�ffC:��ۅ�L��<��
>\)C:��                                    Bx�M/,  �          @޸R��p��0�׾#�
��ffC9�q��p��5���ͿJ=qC9�{                                    Bx�M=�  T          @޸R��p��(��u���RC9���p��!G��B�\���C98R                                    Bx�MLx  �          @�ff����333�.{��Q�C9�����5��G��k�C9�H                                    Bx�M[  T          @���(��G����Ϳ\(�C:s3��(��J=q����\)C:�                                     Bx�Mi�  �          @�ff����:�H����=qC:�����=p���\)�
=qC:�                                    Bx�Mxj  T          @���z�5������\C9�H��z�8Q콸Q�=p�C9�R                                    Bx�M�  �          @�{���Ϳ333��  �33C9Ǯ���Ϳ8Q�8Q�\C9�                                    Bx�M��  T          @�
=��{�(��8Q쿹��C9���{��R���}p�C9�                                    Bx�M�\  �          @�\)�޸R��׾�33�5�C7ٚ�޸R�   �����{C8�                                    Bx�M�  �          @�Q���
=�(���{�/\)C8����
=�#�
��\)��C95�                                    Bx�M��  �          @�G���  �0�׾���C9����  �5�B�\����C9�                                    Bx�M�N  �          @����\)�8Q�8Q쿺�HC9�)��\)�:�H��G��h��C9�R                                    Bx�M��  T          @�  �޸R�5���
��RC9���޸R�5�#�
���
C9ٚ                                    Bx�M�  �          @����\)�B�\���
�(��C:@ ��\)�E��#�
��\)C:J=                                    Bx�M�@  T          @߮��ff�B�\���ͿQ�C:E��ff�E����k�C:Q�                                    Bx�N
�  
�          @޸R���.{����  C9�����.{�u���C9��                                    Bx�N�  T          @޸R��p��0�׾����C9����p��333��\)���C9                                    Bx�N(2  
�          @�p���z��;�=q��RC8�{��z�녾aG����
C8�                                    Bx�N6�  �          @���(���R�#�
���C9���(��!G����ͿL��C98R                                    Bx�NE~  �          @�\)��z�O\)?J=q@У�C:����z�:�H?^�R@�(�C:
=                                    Bx�NT$  �          @�ff��(��0��?+�@�=qC9�q��(���R?=p�@��HC9.                                    Bx�Nb�  �          @������Ϳp��?c�
@陚C;�=���ͿY��?z�HA (�C;�                                    Bx�Nqp  T          @�\)��33�p��?fff@�(�C;�\��33�Y��?z�HA��C;�                                    Bx�N�  
�          @�p���(����>�p�@EC58R��(���G�>\@L(�C4��                                    Bx�N��  T          @�
=��{?����
�8Q�C/����{?���\)���C/�                                     Bx�N�b  <          @�ff��p�>���>���@-p�C1G���p�>�Q�>�z�@�C1�                                    Bx�N�  l          @�ff��p�=�\)?��@�Q�C3s3��p�=�?
=q@�C3�                                    Bx�N��  �          @޸R��=�\)?�@���C3h���=�?�\@�{C2�q                                    Bx�N�T  �          @޸R��>���>�
=@Z�HC1L���>�p�>\@H��C0��                                    Bx�N��  �          @޸R��>�>�@z�HC033��?   >�(�@a�C/�{                                    Bx�N�  �          @޸R��p�>��H?\)@�(�C/�q��p�?
=q?�\@��RC/�=                                    Bx�N�F  T          @�
=��p�>�
=?&ff@��C0�=��p�>�?(�@��C0                                    Bx�O�  �          @�{��(�>�p�?8Q�@�
=C0���(�>�G�?.{@�(�C0T{                                    Bx�O�  �          @�z����
�B�\?�@��C5�����
��?��@�G�C5
                                    Bx�O!8  T          @�z��ۅ�k�?��@��C5���ۅ�.{?�@�\)C5p�                                    Bx�O/�  T          @��
���H>���?
=@��RC15����H>Ǯ?\)@���C0��                                    Bx�O>�  T          @��
�ڏ\?�>�ff@n{C/:��ڏ\?(�>\@Mp�C.�H                                    Bx�OM*  �          @ڏ\��=q>��R>�\)@Q�C1\)��=q>�{>�  @C1!H                                    Bx�O[�  T          @�33��G���G�?+�@��HC7����G���p�?5@�{C7
                                    Bx�Ojv  �          @�(���33��\)?   @�z�C6O\��33�aG�?�@��C5޸                                    Bx�Oy  �          @�p����ͽ�Q�>�=q@G�C4�����ͽu>�\)@�C4xR                                    Bx�O��  "          @�33��ff���
?���A2ffC6���ff�8Q�?���A6{C5��                                    Bx�O�h  T          @��
��ff��?�A>�\C8s3��ff��p�?�(�AD��C7(�                                    Bx�O�  �          @ۅ��p���?�  AJ=qC8���p���p�?�ffAP��C7!H                                    Bx�O��  l          @�ff��(��@  ?��HA�33C:u���(��
=q@�A��C8�H                                    Bx�O�Z  
�          @�\)�ҏ\�W
=@
�HA��RC;G��ҏ\���@\)A�{C9@                                     Bx�O�   �          @�Q����ÿaG�@�A�=qC;�����ÿ�R@��A��
C9k�                                    Bx�Oߦ  �          @�
=�θR�G�@��A���C:���θR��\@!G�A��C8��                                    Bx�O�L  T          @�p��׮?��?���AAG�C/\)�׮?333?���A8(�C.�                                    Bx�O��  
�          @�p����H>8Q�@ffA��C2c����H>��@�
A��HC0k�                                    Bx�P�  T          @޸R��G���  @�A�ffC<����G��B�\@{A��C:��                                    Bx�P>  
�          @�\)�Ӆ�p��@33A�z�C<��Ӆ�5@��A��RC:&f                                    Bx�P(�  �          @�����p��Y��@�\A�
=C;L���p���R@�A���C9W
                                    Bx�P7�  �          @��
�أ׿O\)@�
A���C:�\�أ׿�@��A��C8ٚ                                    Bx�PF0  �          @����ٙ��fff@G�A�p�C;���ٙ��+�@
=A�G�C9�
                                    Bx�PT�  
(          @�����녿aG�?�p�A���C;T{��녿&ff@z�A��RC9p�                                    Bx�Pc|  �          @��
��G��Tz�?�p�A�33C:�3��G����@�
A���C9�                                    Bx�Pr"  T          @�=q��
=�E�@33A�33C:�=��
=��@Q�A�=qC8�=                                    Bx�P��  "          @�{��녿B�\@�A�Q�C:�)��녿�@(�A�\)C8}q                                    Bx�P�n  
�          @����{�:�H@�A�  C:0���{��@(�A��RC8�                                    Bx�P�  T          @�����p��=p�@��A�(�C:�{��p���@p�A�
=C8�                                    Bx�P��  "          @�����&ff@��A�p�C9�������p�@��A�p�C7.                                    Bx�P�`  "          @��أ׿�@
=A�  C8�=�أ׾��
@
=qA�p�C6��                                    Bx�P�  T          @�(����ÿ�@ffA���C8�=���þ��
@	��A�ffC6�R                                    Bx�Pج  �          @�z��ۅ�O\)?�ffAiC:�3�ۅ�
=?��At��C8�                                    Bx�P�R  T          @��H��(��E�?�  AC33C:ff��(��
=?�=qAMC8�                                    Bx�P��  �          @�
=��33��R@2�\A��C9�{��33��z�@5A���C6��                                    Bx�Q�  �          @����Ǯ�.{@7�AĸRC:0��Ǯ����@;�A��C7\                                    Bx�QD  :          @�p�����.{@C33AхC:O\������
@G
=A��C6�                                    Bx�Q!�  �          @�p���Q�(��@7�A�Q�C:���Q쾞�R@;�Aȏ\C6�)                                    Bx�Q0�  
�          @����G��!G�@1G�A�p�C9����G���z�@5�A�p�C6�f                                    Bx�Q?6  
�          @޸R��33�(��@0  A�=qC9�R��33����@3�
A��\C6�R                                    Bx�QM�  �          @�����\)�.{@(Q�A��C9�q��\)��Q�@,(�A�{C7+�                                    Bx�Q\�  <          @�p���(��O\)?��
AfffC:����(��z�?�{Aq��C8ٚ                                    Bx�Qk(  �          @��
�߮��ff?��RA z�C7�3�߮����?��
A&ffC6p�                                    Bx�Qy�  T          @�\�޸R��ff?��HAC7���޸R����?�G�A#�
C6u�                                    Bx�Q�t  �          @�\��
=���H?���A�C8
=��
=��33?�z�A�\C6�f                                    Bx�Q�  
�          @�z���׿+�?�=qA  C9}q��׿
=q?�z�AC8\)                                    Bx�Q��  
�          @����׿E�?�
=AQ�C:G���׿�R?��\A#�C9�                                    Bx�Q�f  �          @�z���׿�?�A33C8�f��׾�
=?�p�A33C7n                                    Bx�Q�  T          @�(��޸R�J=q?��A,��C:p��޸R�(�?�A8Q�C9�                                    Bx�QѲ  T          @�\��z�\(�?�\)A333C;
��z�.{?�(�A?�
C9��                                    Bx�Q�X  �          @�\��z�O\)?�33A6�HC:���z��R?��RAB�HC9.                                    Bx�Q��  
�          @�\���
�O\)?��RAC
=C:�R���
�(�?�=qAO
=C9)                                    Bx�Q��  �          @��H��z�Q�?�
=A9��C:���z�!G�?\AE�C9:�                                    Bx�RJ  �          @����aG�?�=qA,(�C;33���333?�
=A9p�C9                                    Bx�R�  T          @�(��޸R�c�
?�ffA((�C;@ �޸R�5?�33A5��C9�
                                    Bx�R)�  "          @�(���
=�aG�?�  A!G�C;(���
=�5?���A.�\C9��                                    Bx�R8<  �          @��H����aG�?��A*�\C;:�����333?�z�A8  C9�=                                    Bx�RF�  �          @ᙚ��p��@  ?���A  C:33��p����?�Q�A�C8��                                    Bx�RU�  
�          @����p��0��?��\A�C9����p����?���A�C8�=                                    Bx�Rd.  <          @�{��G��E�?�(�A!��C:p���G����?��A-��C9\                                    Bx�Rr�            @����˅���@A�p�C=� �˅�O\)@p�A�ffC;@                                     Bx�R�z  �          @����ə�����@�A�Q�C>� �ə��fff@z�A�=qC<)                                    Bx�R�   �          @����33���@(�A�  C>(���33�W
=@z�A�\)C;��                                    Bx�R��  �          @ڏ\��33���@�A�{C=W
��33�:�H@��A��\C:��                                    Bx�R�l  T          @�=q�˅����@p�A�p�C=�{�˅�E�@A�Q�C:��                                    Bx�R�  T          @�{��ff���@;�AǮC=�R��ff�0��@C33A�ffC:Q�                                    Bx�Rʸ  �          @�
=�ə����@.{A�z�C=�{�ə��8Q�@6ffA�G�C:xR                                    Bx�R�^  �          @�p���p�����@ffA��RC=z���p��=p�@{A�p�C:�)                                    Bx�R�  �          @��������@��A�G�C=\)���=p�@��A�  C:�{                                    Bx�R��  
�          @�����{��{@��A��HC=����{�L��@�A�(�C;
                                    Bx�SP  �          @�ff�θR����@z�A�Q�C=c��θR�=p�@��A�
=C:��                                    Bx�S�  �          @�ff��ff�n{@=qA���C<0���ff�
=@!G�A�{C933                                    Bx�S"�  �          @�
=��{�L��@#33A�  C;���{��G�@(��A�{C7��                                    Bx�S1B  �          @޸R���J=q@!�A��RC:�R����(�@'�A��RC7��                                    Bx�S?�  �          @�
=���p��@!�A�(�C<L����z�@(Q�A���C9�                                    Bx�SN�  T          @�  �Ϯ�0��@!�A�p�C:{�Ϯ����@&ffA��\C6�                                    Bx�S]4  �          @�G���G��@  @\)A�  C:�=��G��Ǯ@$z�A���C7n                                    Bx�Sk�  �          @�\)��G��W
=@G�A��C;L���G���\@
=A�Q�C8s3                                    Bx�Sz�  T          @�{�θR�W
=@��A��C;^��θR���H@\)A�(�C8Q�                                    Bx�S�&  �          @�ff��
=�^�R@�A���C;����
=��@{A���C8�)                                    Bx�S��  �          @޸R��Q�n{@�\A�C<)��Q�
=@��A�p�C9.                                    Bx�S�r  �          @�  ��녿c�
@G�A�p�C;�q��녿\)@Q�A���C8�
                                    Bx�S�  �          @�
=�Ϯ�Q�@��A�=qC;(��Ϯ��@�RA���C8{                                    Bx�Sþ  �          @�����p��E�@�HA��
C:�)��p����@   A�  C7�3                                    Bx�S�d  �          @޸R��p����@{A�  C=����p��8Q�@&ffA�\)C:k�                                    Bx�S�
  �          @޸R�˅����@'
=A�(�C=�=�˅�.{@/\)A�G�C:�                                    Bx�S�  �          @�ff��p��xQ�@�RA�G�C<����p����@&ffA�p�C9Q�                                    Bx�S�V  �          @�
=���H����@,��A�=qC=�\���H�(��@5�A�p�C9�                                    Bx�T�  T          @���˅���@0��A�
=C=}q�˅�#�
@8Q�A�(�C9Ǯ                                    Bx�T�  �          @�  �ʏ\��ff@2�\A��C=s3�ʏ\�!G�@:=qA��HC9��                                    Bx�T*H  �          @�Q���p��fff@*=qA�z�C<  ��p��   @1G�A��C8h�                                    Bx�T8�  
�          @�ff�Ϯ�aG�@�
A�
=C;���Ϯ��@=qA�z�C8�H                                    Bx�TG�  �          @�ff���aG�@\)A�=qC;�
���   @&ffA��C8s3                                    Bx�TV:  
�          @�\)��
=�^�R@p�A�G�C;����
=���H@$z�A���C8Y�                                    Bx�Td�  "          @������ÿxQ�@=qA��RC<xR���ÿ��@!�A�
=C95�                                    Bx�Ts�  �          @���θR���@!G�A�=qC=+��θR�&ff@)��A�\)C9��                                    Bx�T�,  
�          @����
=���
@   A�G�C=
=��
=�!G�@(Q�A�=qC9��                                    Bx�T��  
�          @�
=���Ϳ�G�@#�
A���C<�����Ϳ��@,(�A�p�C9c�                                    Bx�T�x  �          @���z�fff@!G�A���C<\��z��\@(Q�A��\C8�                                    Bx�T�  T          @������
�O\)@\)A��C;J=���
��
=@%A��\C7�                                    Bx�T��  T          @��˅�Q�@(��A�33C;O\�˅�Ǯ@/\)A�  C7�\                                    Bx�T�j  T          @�ff�θR�xQ�@Q�A�(�C<���θR�z�@   A��RC9+�                                    Bx�T�  
�          @�
=��\)����@z�A��C=�
��\)�8Q�@p�A��C:T{                                    Bx�T�  �          @�ff��ff��{@A�G�C=�R��ff�8Q�@�RA��C:ff                                    Bx�T�\  �          @���{��  @
=A�\)C<�\��{�(�@\)A�ffC9p�                                    Bx�U  �          @�Q��У׿�G�@�A�C<�
�У׿!G�@   A��HC9}q                                    Bx�U�  
�          @�����
=���@
=qA�=qC=O\��
=�333@�
A�(�C:33                                    Bx�U#N  T          @�(���z�B�\@=qA�z�C:Ǯ��zᾸQ�@   A���C7B�                                    Bx�U1�  "          @�{���Ϳ�R@&ffA�Q�C9����;B�\@*�HA���C5�R                                    Bx�U@�  �          @�ff��  �Tz�@�
A��C;L���  ��ff@�HA���C7��                                    Bx�UO@  �          @�Q���녿G�@A�ffC:����녾Ǯ@(�A�
=C7n                                    Bx�U]�  �          @���љ��\)@��A�C8�f�љ��#�
@ ��A��
C5^�                                    Bx�Ul�  �          @����G��\@'
=A��C7Y���G�=L��@(Q�A�p�C3�
                                    Bx�U{2  �          @�  ��{���R@.�RA��C6����{>\)@0  A���C2Ǯ                                    Bx�U��  �          @�
=��zᾮ{@1G�A�
=C7\��z�=�@2�\A�ffC2�3                                    Bx�U�~  �          @�  ������R@333A�=qC6�\���>��@3�
A�33C2�f                                    Bx�U�$  �          @������G�@2�\A�G�C7�3��<��
@5�A�C3�=                                    Bx�U��  �          @߮��z�
=@0��A�=qC9B���z��@4z�A�z�C5
                                    Bx�U�p  �          @�
=�˅�#�
@1G�A�p�C9Ǯ�˅�.{@5A�ffC5�\                                    Bx�U�  �          @�
=���H�:�H@2�\A��C:�����H��  @8Q�A�
=C6G�                                    Bx�U�  �          @�Q���(���G�@.{A��HC=��(��
=q@7
=A�=qC8�
                                    Bx�U�b  �          @�����zῊ=q@-p�A�C=�H��z�(�@7
=A�  C9u�                                    Bx�U�  �          @�=q�����R@*�HA�p�C>�����E�@5A�p�C:��                                    Bx�V�  �          @�\�Ϯ�h��@*�HA�
=C;��Ϯ��G�@1�A��C7޸                                    Bx�VT  T          @���θR�.{@1G�A��RC:��θR�L��@5A�{C5�                                    Bx�V*�  �          @�\��
=�G�@1G�A�z�C:����
=����@7�A�
=C6�H                                    Bx�V9�  �          @����Q�#�
@:=qA��C9�
��Q��@>{A�ffC5�                                    Bx�VHF  n          @�z����
=q@C33Aʏ\C8��    @FffA��
C4                                      Bx�VV�  :          @�(���\)�
=@9��A��C9:���\)���
@<��A�{C4��                                    Bx�Ve�  
�          @�=q���#�
@8Q�A�(�C9�3����@<��A�
=C5(�                                    Bx�Vt8  T          @��ʏ\��(�@L��A�ffC7�)�ʏ\>\)@N�RA�=qC2�q                                    Bx�V��  �          @�=q��33��@C�
A���C8����33<#�
@FffA�(�C3�f                                    Bx�V��  
�          @����ff���@UA��C7����ff>8Q�@W
=A�p�C2Q�                                    Bx�V�*  �          @�=q��녿�@G�A��C8� ���=#�
@J=qA��C3�                                    Bx�V��  �          @�=q���H��@E�AΏ\C9!H���H�#�
@HQ�A�Q�C4�                                    Bx�V�v  "          @ᙚ���
�.{@<��A�{C:!H���
���@A�A�p�C5Q�                                    Bx�V�  �          @�  �ʏ\�8Q�@:=qA�=qC:���ʏ\�B�\@?\)A�(�C5��                                    Bx�V��  �          @߮��p���@Mp�A���C9O\��p�<#�
@P��A޸RC3�f                                    Bx�V�h  �          @޸R�ə��\(�@4z�A�
=C;���ə�����@:�HA��HC7
=                                    Bx�V�  �          @�����+�@.{A�=qC9�R���.{@2�\A��C5�                                    Bx�W�  �          @ᙚ�Ϯ�+�@*=qA�33C9ٚ�Ϯ�.{@.�RA���C5�                                    Bx�WZ  �          @����  �z�@*�HA��C9)��  ��Q�@.�RA�=qC4�                                     Bx�W$   <          @�=q�����
=@<��AŅC7�R���>�@>�RA�p�C2�
                                    Bx�W2�  
�          @߮��ff�}p�@�RA��\C<����ff��@'�A�Q�C8��                                    Bx�WAL  
�          @޸R��ff�c�
@��A��C;����ff��(�@%�A�C7�=                                    Bx�WO�  �          @߮��\)�u@p�A���C<k���\)���H@%A�{C8O\                                    Bx�W^�  
�          @�  ��
=�z�H@   A���C<����
=�   @(Q�A��RC8s3                                    Bx�Wm>  �          @����
=��  @ ��A�C<����
=��@)��A��
C8�
                                    Bx�W{�  �          @�=q�θR�u@,��A���C<z��θR��G�@5�A��HC7�                                    Bx�W��  T          @�Q���G��Y��@>�RA�\)C;����G���\)@EA��C6��                                    Bx�W�0  "          @�ff��Q�p��@7�A�\)C<� ��Q�\@@  A�ffC7u�                                    Bx�W��  T          @�
=��
=�Y��@A�A�ffC;����
=���@H��A�=qC6n                                    Bx�W�|  T          @�Q����ÿ�  @<��A�\)C=\���þ�
=@EA�33C7�)                                    Bx�W�"  �          @ᙚ��녿��H@8Q�A�33C>����녿#�
@C�
A�  C9�{                                    Bx�W��  T          @�(���(���@=p�Aď\C>k���(��
=@HQ�AЏ\C9@                                     Bx�W�n  �          @�����n{@EA�{C<=q������
@Mp�Aԣ�C6ٚ                                    Bx�W�  �          @���ə���(�@XQ�A�33C7�H�ə�>k�@Y��A��C1��                                    Bx�W��  �          @�(���{��\)@a�A��C4����{?(�@^�RA�C.W
                                    Bx�X`  �          @�
=�ʏ\����@]p�A�
=C6���ʏ\>�33@]p�A��HC0ٚ                                    Bx�X  �          @�����Q쾨��@\��A���C7���Q�>�33@\��A���C0�{                                    Bx�X+�  �          @��
�ȣ׾�@VffA�=qC85��ȣ�>L��@W�A�(�C2.                                    Bx�X:R  �          @�����G��z�@UA޸RC9B���G�=��
@X��A�Q�C3=q                                    Bx�XH�  �          @�z��ȣ׿(��@UA�
=C:��ȣ�<#�
@Y��A�C3�                                    Bx�XW�  "          @�p���z�8Q�@K�Aң�C:xR��zὣ�
@P��A�ffC4�                                     Bx�XfD  T          @�{��{�B�\@FffA�z�C:�R��{��@L(�A���C5+�                                    Bx�Xt�  �          @�
=��ff�5@J=qAυC:G���ff��\)@N�RA�
=C4��                                    Bx�X��  �          @�R���L��@J=qA�C;
���#�
@P  A֏\C5c�                                    Bx�X�6  �          @�ff��ff�aG�@E�A�ffC;� ��ff��  @L(�A�ffC6.                                    Bx�X��  T          @�p�����}p�@C�
A�C<Ǯ�����Q�@L(�A�p�C733                                    Bx�X��  T          @��H���H���@?\)AǮC=����H��G�@I��Aң�C7��                                    Bx�X�(  �          @�z����Ϳ�{@5A��C@���ͿB�\@C�
A���C:�=                                    Bx�X��  �          @�
=��{����@;�A�\)C@����{�Tz�@J=qAϮC;L�                                    Bx�X�t  �          @����Ϳ�\)@4z�A��CB=q���Ϳ��\@EA�(�C=
=                                    Bx�X�  "          @���p���(�@7
=A��C@���p��\(�@FffA�ffC;��                                    Bx�X��  �          @�{�����@=p�A�(�C?^����(��@J=qA�ffC9޸                                    Bx�Yf  �          @������Ϳ�
=@=p�A��
C>p����Ϳ��@H��A�z�C8�H                                    Bx�Y  �          @������
��33@AG�A�C>5����
�   @L(�A�  C8��                                    Bx�Y$�  �          @����
�z�H@J=qA��HC<�R���
����@R�\A�ffC6��                                    Bx�Y3X  �          @�R��33�c�
@Q�A؏\C;����33�B�\@X��A�RC5��                                    Bx�YA�  
Z          @�ff���ÿG�@[�A��
C;����ýL��@aG�A�(�C4k�                                    Bx�YP�  T          @��ʏ\�G�@Z=qA�
=C;��ʏ\�L��@`  A�\)C4u�                                    Bx�Y_J  "          @�ff�Ǯ�#�
@a�A���C9���Ǯ=�@e�A���C2�                                    Bx�Ym�  T          @�ff��p���\@k�A�C8���p�>�=q@mp�A��C1�                                     Bx�Y|�  "          @����Ϳ\)@j=qA�p�C9(�����>k�@l��A�{C1޸                                    Bx�Y�<  T          @�p���ff���@c33A�G�C9�=��ff>#�
@fffA���C2��                                    Bx�Y��  �          @�{��ff�#�
@e�A���C9����ff>�@hQ�A���C2�=                                    Bx�Y��  �          @�ff�Ǯ�B�\@`  A�
=C:���Ǯ    @e�A�
=C3�q                                    Bx�Y�.  �          @�����  �W
=@XQ�A�\)C;����  ����@^�RA�RC4�H                                    Bx�Y��  
�          @�z���  �Tz�@S�
Aݙ�C;�{��  ����@Z=qA�
=C4�3                                    Bx�Y�z  �          @��
�ʏ\���@C33A�
=C=�ʏ\���@Mp�AָRC7�3                                    Bx�Y�   "          @��
�ə��Y��@L(�A�33C;�R�ə����@S33A��C5T{                                    Bx�Y��  �          @������xQ�@B�\A�33C<�������z�@K�A�
=C6��                                    Bx�Z l  �          @�{�Ϯ��  @;�A�=qC<�R�Ϯ��{@Dz�A�z�C6�R                                    Bx�Z  �          @�����H�z�H@'�A��C<p����H�Ǯ@1G�A��C7Y�                                    Bx�Z�  �          @�����
�xQ�@#33A��RC<^����
�Ǯ@,��A��C7h�                                    Bx�Z,^  �          @�33���H�Tz�@ ��A��C;+����H���@(Q�A��C6B�                                    Bx�Z;  �          @��H����Tz�@�A��C;�������R@��A��\C6�f                                    Bx�ZI�  
�          @�����{�L��@�A�
=C:�{��{���@\)A��C68R                                    Bx�ZXP  �          @���{�Y��@7�A�z�C;z���{�B�\@>�RAƣ�C5�3                                    Bx�Zf�  T          @�z����5@]p�A�Q�C:�
��=�\)@a�A��C3Y�                                    Bx�Zu�  T          @���
=��R@s33B\)C9�3��
=>u@vffB  C1��                                    Bx�Z�B  �          @�\���\�+�@}p�B�C:�\���\>k�@�  B	�HC1�q                                    Bx�Z��  "          @�(���{��R@y��B��C9���{>�=q@|(�B{C1^�                                    Bx�Z��  T          @����;�@|��B�HC8}q����>�G�@|��B  C/Ǯ                                    Bx�Z�4  �          @�\���H��Q�@��B#G�C4����H?s33@���B{C)޸                                    Bx�Z��  �          @������>�(�@��B133C/����?�  @��HB'(�C#                                      Bx�Z̀  �          @�p����>��@�Q�B.{C/O\���?�p�@�G�B$\)C#�                                    Bx�Z�&  �          @�ff��  ���
@���B#�RC4=q��  ?�G�@�{B��C)�                                    Bx�Z��  �          @�
=��G�    @�G�B"ffC4���G�?��
@�p�BG�C(�3                                    Bx�Z�r  "          @�G���\)=�G�@��B+(�C2��\)?�z�@��RB$z�C&��                                    Bx�[  T          @�  ���H>aG�@�{B<G�C1G����H?���@�Q�B3ffC#��                                    Bx�[�  �          @�G�����>B�\@��\B5\)C1�3����?�ff@���B-(�C$�                                    Bx�[%d  �          @�����G�=�G�@���B4{C2����G�?��H@�z�B,�
C%�{                                    Bx�[4
  �          @�33����>�@���B7
=C2k�����?�G�@��B/Q�C%+�                                    Bx�[B�  
�          @�p����\>aG�@��RB6�HC1Y����\?�\)@���B.(�C$.                                    Bx�[QV  �          @�ff���>aG�@��\B;��C1L����?�33@�z�B2�HC#�                                    Bx�[_�  "          @������?&ff@�ffBQ�C+{����?�z�@���BA�CQ�                                    Bx�[n�  �          @ۅ��  ?�ff@�\)BS�
C%@ ��  @�
@��\B>��C�                                    Bx�[}H  �          @ۅ���\?�  @��BP�RC&0����\@  @���B<�C(�                                    Bx�[��  
�          @�33����?!G�@��BP�HC+L�����?��@��BA{CxR                                    Bx�[��  T          @ۅ���?\)@��BM�C,�=���?�@��\B?{C�                                    Bx�[�:  �          @���\)?��
@�  BT�\C%z��\)@�
@�33B?z�C��                                    Bx�[��  �          @�ff�~�R?��
@��HBV��C%���~�R@�@�{BA\)C�3                                    Bx�[Ɔ  T          @�{���?��@�33BWG�C,O\���?�\)@��BG��CJ=                                    Bx�[�,  
�          @�����>L��@��BH�C1n���?��H@�33B>C!��                                    Bx�[��  �          @�(���\)=u@��RBE\)C3E��\)?��@���B={C#�q                                    Bx�[�x  T          @�p���Q�W
=@�Q�B9�RC6�\��Q�?�  @�p�B5(�C(&f                                    Bx�\  �          @�p���{��  @��HB1z�C6�3��{?k�@�Q�B-�RC)s3                                    Bx�\�  
�          @��������
@�33B{C7h����?=p�@���B�HC,)                                    Bx�\j  T          @����(��:�H@z�HB
�C;h���(�>u@~�RB��C1��                                    Bx�\-  �          @���(��s33@E�A�G�C<Ǯ��(��.{@N{A݅C5��                                    Bx�\;�  �          @�\��G���{?�
=A��CAT{��G����\?��AH��C>�)                                    Bx�\J\  
�          @���=q���?5@�  CC���=q��z�?�33Ap�CA��                                    Bx�\Y  
Z          @��
�������?&ff@��RCA\������?��\A�
C?h�                                    Bx�\g�  "          @�(���p���?   @���CA�{��p���  ?fff@�\)C@5�                                    Bx�\vN  T          @�z���ff��(�?333@�z�C?���ff��  ?��AffC>0�                                    Bx�\��  "          @�����\?+�@�p�C@p�������?��
A�C>�q                                    Bx�\��  �          @�=q��=q��ff>�@p��CB���=q�У�?fff@���CAh�                                    Bx�\�@  T          @��H�أ׿�\)?k�@�ffCCxR�أ׿˅?�{A0z�CA0�                                    Bx�\��  
�          @ᙚ�ָR���
?��RAC�
C@�\�ָR��{?�=qAp��C=Y�                                    Bx�\��  T          @޸R��ff���@3�
A��\C>k���ff����@@��A��HC7�                                    Bx�\�2  "          @��
��ff����@,��A�{C=�
��ff��Q�@8��AǅC7W
                                    Bx�\��  �          @�33��z῝p�@	��A�ffC>�H��z�&ff@��A�p�C9�=                                    Bx�\�~  T          @�������H?�(�ADz�C>E����L��?�p�Ag\)C:��                                    Bx�\�$  �          @�����p���  ?��\A	�C>����p��s33?��A/�C<!H                                    Bx�]�  
�          @�\)�ָR��=q��Q��>�RCC=q�ָR��{=�G�?n{CC�                                     Bx�]p  	�          @�{�����{���}p�CC�\�����<#�
=�\)CD�                                    Bx�]&  �          @޸R��{��=q����X��CCQ���{���=�\)?z�CC��                                    Bx�]4�  
�          @�ff��\)���#�
��=qCA���\)��z�>�=q@\)CA�\                                    Bx�]Cb  
�          @���
=��z�����CA����
=���>���@�RCA��                                    Bx�]R  �          @�Q���Q���
����ffCB�R��Q��  >���@.�RCB�                                     Bx�]`�  �          @�=q���H���
�u��Q�CB�����H���
>k�?�CB�
                                    Bx�]oT  T          @�  ��녿�z�����(�CA����녿��>�z�@CA��                                    Bx�]}�  T          @�{��Q��=q�L�;�
=CA33��Q���>�Q�@?\)C@�)                                    Bx�]��  
t          @�ff�׮���u����CA���׮��>W
=?ٙ�CA�                                    Bx�]�F  
�          @�\)�ٙ�����W
=��G�C@� �ٙ����>B�\?���C@�                                    Bx�]��  T          @���  ���R���
�'
=C@u���  �\=�Q�?333C@��                                    Bx�]��  �          @�=q��z�\��p��E�C@�f��z�Ǯ=#�
>\CA@                                     Bx�]�8  T          @ٙ��Ӆ���þ�=q��CAaH�Ӆ�˅>��?��
CA��                                    Bx�]��  
�          @�=q���
��\)��  ��CA�R���
�У�>B�\?���CA�=                                    Bx�]�  T          @�ff��{�\���\��C@�=��{��p��
=���HCB�=                                    Bx�]�*  "          @�\)���ÿ�Q�� ����  C@h����ÿ��Ǯ�N�\CD^�                                    Bx�^�  �          @߮�љ���Q���H��Q�C@aH�љ���z��G��G\)CD8R                                    Bx�^v  �          @޸R��ff��
=��(��C
=C=�R��ff���
������C@�
                                    Bx�^  "          @߮��{����\�I��C>�3��{��33��\)�(�CA�H                                    Bx�^-�  �          @�\)���Ϳ��Ϳ�33�Z�\C?u����Ϳ޸R��p��"{CB��                                    Bx�^<h  �          @�����H��Q��(��aG�C?�H���H��  ���k�C@aH                                    Bx�^K  "          @�{��{��G�?z�HAC@���{��Q�?���A333C>\                                    Bx�^Y�  "          @�z������?�G�A33C@  ������?�{A5C=L�                                    Bx�^hZ  
�          @�����  ��G�?(�@��C>�
��  ��ff?n{@�
=C<�
                                    Bx�^w   �          @�z�����z�?c�
@��C?������{?��RA%p�C=xR                                    Bx�^��  T          @�����33��G�?�p�A$��C@����33��\)?���AVffC=��                                    Bx�^�L  �          @ۅ����\)>�
=@_\)C?�{������?E�@�  C>8R                                    Bx�^��  
�          @����G���33@   A�\)C@����G��(��@1�A��C:                                    Bx�^��  
�          @�33���ÿ��R@�A���CA\)���ÿO\)@&ffA���C;\)                                    Bx�^�>  �          @ۅ������H?�z�A���C@�\����aG�@\)A�z�C;��                                    Bx�^��  �          @ۅ�Ϯ����?�(�AECA�H�Ϯ���?���Az�HC=��                                    Bx�^݊  T          @ڏ\��(���33@   A�(�C@\)��(��J=q@�
A��C;{                                    Bx�^�0  �          @�=q�ȣ׿���@33A�C@0��ȣ׿+�@%�A�=qC:�                                    Bx�^��  "          @أ���
=��z�@�\A��
C@���
=�8Q�@%A��C:��                                    Bx�_	|  �          @ٙ��ə�����@
=qA�(�C@c��ə��:�H@p�A���C:�)                                    Bx�_"  T          @�=q��z῱�?��HA��
C@@ ��z�J=q@G�A���C;�                                    Bx�_&�  T          @��
��Q쿳33?�Q�Ad  C@���Q�^�R@ ��A�ffC;��                                    Bx�_5n  T          @ۅ��Q쿵?��A\��C@J=��Q�fff?�(�A�p�C;��                                    Bx�_D  T          @������ÿ�Q�?�A`Q�C@s3���ÿk�@ ��A�p�C;�q                                    Bx�_R�  
�          @��H���Ϳ�p�?�\)A~ffCA
=���Ϳfff@p�A�33C;��                                    Bx�_a`  �          @��H�ə���  @(�A�G�CAc��ə��Q�@!G�A��C;p�                                    Bx�_p  �          @ۅ��(�����@�
A�{C@�=��(��O\)@��A�
=C;5�                                    Bx�_~�  T          @߮��zΌ��@�A���C@�=��z�333@/\)A��HC:G�                                    Bx�_�R  �          @�=q�����H@!�A��
C@����.{@5A��C:
=                                    Bx�_��  
�          @����z�\@%A�z�CAff��z�8Q�@:�HA�p�C:s3                                    Bx�_��  �          @����{���R@�RA�z�CA  ��{�8Q�@333A���C:aH                                    Bx�_�D  �          @ᙚ�θR���
@�A��CAQ��θR�J=q@-p�A��RC:�R                                    Bx�_��  �          @��H��\)����@p�A�ffC@�)��\)�0��@1G�A�(�C:
=                                    Bx�_֐  T          @����ÿ��H@��A�  C@�����ÿ5@.{A��C:33                                    Bx�_�6  
�          @��У׿��R@=qA��RC@ٚ�У׿:�H@/\)A�G�C:ff                                    Bx�_��  
�          @�\���ÿ�
=@z�A�33C@^����ÿ333@(��A���C:#�                                    Bx�`�  T          @�33��G�����@A�C@����G��8Q�@*=qA��C:B�                                    Bx�`(  
Z          @��
��G���
=@�HA���C@L���G��+�@.�RA�(�C9��                                    Bx�`�  �          @�(���=q��G�@z�A�=qC@���=q�E�@*=qA�p�C:��                                    Bx�`.t  �          @�33��=q��(�@  A��C@�)��=q�@  @%�A�ffC:��                                    Bx�`=  �          @�=q�ҏ\��
=@	��A��
C@5��ҏ\�=p�@�RA��C:aH                                    Bx�`K�  
�          @����Q���
@z�A�ffCCO\��Q쿌��@   A��C=��                                    Bx�`Zf  T          @ᙚ��
=��{@�RA��
CB  ��
=�c�
@&ffA��
C;�
                                    Bx�`i  T          @�=q��z��=q@"�\A���CA�H��z�E�@8��A�p�C:�{                                    Bx�`w�  
Z          @�\��\)��33@33A���CBO\��\)�h��@+�A�=qC;�3                                    Bx�`�X  �          @����p���33@(�A���CBs3��p��\(�@4z�A�(�C;��                                    Bx�`��  �          @��
��ff��@   A�=qCB���ff�\(�@8Q�A��HC;�
                                    Bx�`��  T          @��
��ff�ٙ�@\)A�CB�=��ff�c�
@8Q�A�
=C;�)                                    Bx�`�J  "          @����Ϯ��
=@\)A�
=CB���Ϯ�^�R@8Q�A�C;��                                    Bx�`��  "          @�����  ��Q�@��A��\CB�=��  �aG�@6ffA��C;�R                                    Bx�`ϖ  T          @�(���
=���@ ��A���CBE��
=�Q�@8��A�
=C;@                                     Bx�`�<  T          @��
��ff��(�@   A�(�CB����ff�c�
@9��A��
C;��                                    Bx�`��  
�          @�=q�����z�@p�A�G�CB�=����Y��@6ffA�=qC;�\                                    Bx�`��  �          @ᙚ��{��\)@�A���CB���{�Tz�@/\)A�33C;aH                                    Bx�a
.  
�          @ᙚ��ff���@�A�ffCBG���ff�^�R@.{A�33C;�f                                    Bx�a�  
Z          @��H��  ���@�\A�
=CB+���  �aG�@+�A�C;��                                    Bx�a'z  �          @�33�У׿�@��A�Q�CBW
�У׿h��@*=qA��C;�                                    Bx�a6   
�          @�=q�Ϯ��
=@��A���CB� �Ϯ�k�@*=qA��RC<�                                    Bx�aD�  �          @��H��Q��z�@  A�{CBE��Q�fff@)��A�\)C;ٚ                                    Bx�aSl  
�          @�(���녿�\)@G�A�ffCA޸��녿\(�@)��A���C;n                                    Bx�ab  
�          @�����\)���
@��A�(�CAE��\)�E�@'�A���C:                                    Bx�ap�  �          @�Q��θR����@p�A�
=CA�H�θR�Y��@%A��C;xR                                    Bx�a^  
�          @�Q���ff��{@{A��CA����ff�Y��@&ffA�Q�C;��                                    Bx�a�  
�          @ᙚ��\)����@G�A�z�CA�{��\)�L��@(��A�(�C;                                      Bx�a��  �          @�=q��Q��Q�@�A��
CB�=��Q�p��@%A�=qC<5�                                    Bx�a�P  �          @ᙚ��\)��\@	��A�=qCCB���\)���\@%A��\C<�3                                    Bx�a��  �          @�\��  ��p�@��A���CB�)��  �u@'�A�  C<p�                                    Bx�aȜ  �          @ᙚ��\)��p�@	��A��
CB����\)�}p�@%�A��C<�f                                    Bx�a�B  �          @�\��Q�޸R@
�HA��\CB�3��Q�z�H@&ffA�=qC<��                                    Bx�a��  
Z          @����  �ٙ�@�A��
CB���  �p��@&ffA���C<E                                    Bx�a�  T          @����  �ٙ�@�A�  CB����  �p��@&ffA�
=C<B�                                    Bx�b4  T          @����Q���H@�A�CB� ��Q�xQ�@#33A�33C<z�                                    Bx�b�  �          @ᙚ���ÿ�
=@�A�{CBu����ÿxQ�@p�A�33C<n                                    Bx�b �  "          @���ҏ\��\)?�p�A��\CAٚ�ҏ\�k�@��A�Q�C;��                                    Bx�b/&  	�          @ᙚ�Ӆ��\)?���As�CA�Ӆ�u@G�A��C<@                                     Bx�b=�  
(          @�\��z��\)?�{As33CA��z�u@G�A��C<=q                                    Bx�bLr  �          @ᙚ�Ӆ���?�Am�CA���Ӆ�}p�@�RA���C<�                                     Bx�b[  T          @�{�У׿���?��
Am�CA�У׿u@(�A���C<\)                                    Bx�bi�  "          @ٙ����Ϳ��
?��HAh��CAz����Ϳk�@ffA�(�C<(�                                    Bx�bxd  
�          @�  ���H���
?�  Ao�
CA�
���H�fff@��A��C<)                                    Bx�b�
  T          @�ff��G���=q?�
=AhQ�CB���G��xQ�@A��C<�                                    Bx�b��  �          @�\)��33����?�ffAT��CB+���33���
?�p�A���C=8R                                    Bx�b�V  �          @�ff��G����R?�\At��CAW
��G��Y��@	��A��
C;�R                                    Bx�b��  "          @�p��Ǯ��{?�A��C@O\�Ǯ�.{@��A�{C:8R                                    Bx�b��  �          @�ff��녿��?��Axz�C?����녿333@Q�A�  C:J=                                    Bx�b�H  T          @׮���H���\?��A�{C?T{���H���@(�A�  C9n                                    Bx�b��  �          @�\)���H��  ?�\)A�
=C?#����H�
=@
�HA��\C9G�                                    Bx�b�  
�          @ָR���H��p�?�AzffC>�q���H�
=@
=A��\C9J=                                    Bx�b�:  
Z          @�ff��33���?�33Adz�C?����33�=p�?��RA���C:�H                                    Bx�c
�  T          @�  ��p�����?���AZ�RC?ٚ��p��B�\?�Q�A��C:�                                    Bx�c�  T          @׮��{��ff?�  A*�HCA� ��{���?�
=Ag
=C=\)                                    Bx�c(,  "          @����θR�޸R?��A
=CC\�θR���
?��A`(�C?.                                    Bx�c6�  T          @�\)���Ϳ��
?���A>{CA}q���Ϳ�  ?�AyG�C<�                                    Bx�cEx  
(          @ڏ\��Q��G�?��A<  CA
=��Q�z�H?�ffAu�C<��                                    Bx�cT  �          @�(���녿�ff?���A8��CAG���녿��\?�ffAs33C<��                                    Bx�cb�  �          @�(��љ����?�
=A?�CA8R�љ��}p�?���Ay��C<�)                                    Bx�cqj  T          @����  ���?��RAJ�HC@���  �Q�?�{A~=qC;=q                                    Bx�c�  "          @��
��녿�z�?�
=A@  C@���녿\(�?�AtQ�C;�                                     Bx�c��  "          @�z���33����?�A>�RC?�{��33�O\)?��Ap��C;�                                    Bx�c�\  
�          @���zΌ��?��A8z�C@J=��z�h��?��
AnffC;�
                                    Bx�c�  
Z          @����Ϳ�Q�?��A2ffC@8R���Ϳk�?޸RAhQ�C;��                                    Bx�c��  
�          @����(�����?�{A5p�C?� ��(��\(�?޸RAh��C;\)                                    Bx�c�N  �          @�(��Ӆ���?���A0��C?�)�Ӆ�aG�?ٙ�AeG�C;��                                    Bx�c��  T          @ۅ���H���?�G�A(��C?�����H�fff?��A^{C;�                                    Bx�c�  T          @��H�љ���Q�?���A1C@c��љ��k�?�(�Ah��C<                                    Bx�c�@  �          @�����
���H?���A/�
C@k����
�p��?�p�Ag\)C<
                                    Bx�d�  T          @߮��\)���?�G�A&ffC?����\)�fff?�33AZ�RC;�
                                    Bx�d�  
�          @�G���G�����?��HAC?� ��G��fff?���AQC;�{                                    Bx�d!2  
�          @����Q쿸Q�?��HA\)C?�q��Q�s33?�\)AUC<�                                    Bx�d/�  �          @�
=��ff����?��RA#�C@5���ff�u?�33A[
=C<�                                    Bx�d>~  �          @�Q��׮��Q�?�  A$��C@{�׮�p��?�z�A[�C;��                                    Bx�dM$  
�          @�
=�׮��33?��A=qC?���׮�p��?��AL  C;�                                    Bx�d[�  �          @޸R�أ׿�\)?Y��@�\)C?}q�أ׿�G�?�G�A&�\C<��                                    Bx�djp  �          @���  ���?G�@θRC?@ ��  ��  ?�
=Ap�C<u�                                    Bx�dy  
�          @��׮���?Y��@�\C?���׮���\?��\A)�C<�f                                    Bx�d��  
�          @޸R�أ׿�{?Tz�@�=qC?aH�أ׿�G�?��RA#�C<s3                                    Bx�d�b  �          @�ff���ÿ��?8Q�@�\)C>�R���ÿ}p�?�\)A��C<W
                                    Bx�d�  �          @�{��\)��?fff@�RC?���\)���
?�=qA0z�C<�q                                    Bx�d��  
�          @�\)��
=��ff?��A	p�C@����
=����?�  AG
=C=T{                                    Bx�d�T  T          @�  �أ׿�p�?z�HA ��C@Y��أ׿��?�A;�C<�                                    Bx�d��  T          @�\)�׮��G�?��\A�C@�)�׮����?�(�AB{C=\                                    Bx�dߠ  	�          @�
=��\)���
?xQ�@�\)C@���\)��{?�
=A=�C=T{                                    Bx�d�F  �          @�\)��\)�˅?k�@�CAG���\)��
=?�z�A:ffC=�                                    Bx�d��  T          @�\)�׮��=q?\(�@�33CA&f�׮��Q�?���A2{C=�q                                    Bx�e�  "          @����ff��Q�?h��@��HC@!H��ff���?���A4  C<ٚ                                    Bx�e8  "          @���ָR��
=?Tz�@�{C@��ָR���?��\A)�C<�q                                    Bx�e(�  
�          @����
=��33?G�@θRC?����
=��ff?��HA!p�C<��                                    Bx�e7�  "          @��H��zῸQ�?Q�@�33C@=q��zῈ��?�G�A)C=0�                                    Bx�eF*  �          @ڏ\��(�����?J=q@���C@W
��(����?��RA'\)C=W
                                    Bx�eT�  �          @�{��
=���?B�\@�Q�C@����
=��Q�?��RA$��C>                                      Bx�ecv  
�          @�z�����  ?=p�@�ffC@�����z�?�(�A"�RC=�\                                    Bx�er  
�          @��
��(����
?uA (�CA��(���{?�
=A?�
C=�                                     Bx�e��  �          @ۅ�Ӆ����?fff@�\CA�{�Ӆ��Q�?�33A<Q�C>.                                    Bx�e�h  
�          @�33��33��ff?n{@��HCA33��33����?�z�A>ffC=��                                    Bx�e�  "          @ۅ���
��ff?^�R@陚CA33���
��z�?���A6=qC=�f                                    Bx�e��  
�          @ڏ\��녿��?p��@��RCB\��녿��H?��HAD��C>z�                                    Bx�e�Z  
�          @�33�Ӆ�Ǯ?fff@�CAT{�Ӆ��z�?��A:�HC=�                                    Bx�e�   �          @��H��33��=q?^�R@�\CA� ��33��
=?�\)A8��C>&f                                    Bx�eئ  �          @�(����
��
=?L��@�z�CB@ ���
��ff?�=qA2�\C?�                                    Bx�e�L  T          @�33���H��
=?W
=@ᙚCBT{���H���?�\)A9�C?
=                                    Bx�e��  
�          @��
�Ӆ���
?&ff@��RCC��Ӆ��Q�?�p�A%p�C@G�                                    Bx�f�  "          @�z����
��  ?333@��\CB�����
���?��\A)��C?��                                    Bx�f>  �          @�����z����>Ǯ@N�RCC\)��z����?�  ACAQ�                                    Bx�f!�  
�          @�ff�����>��
@(��CC�\����z�?u@��RCA�3                                    Bx�f0�  
�          @����
=��=�Q�?=p�CD����
=��\)?L��@У�CC��                                    Bx�f?0  S          @�����  ��z�>���@�CC���  ��
=?s33@�Q�CA�R                                    Bx�fM�  �          @�=q��=q��\?�@��RCB�\��=q��p�?�{Az�C@5�                                    Bx�f\|  T          @��ٙ�����?0��@��CD��ٙ���=q?��A-p�CA
                                    Bx�fk"  "          @��H�أ׿�
=?Q�@ӅCC��أ׿\?���A<(�C@�f                                    Bx�fy�  �          @��
�ٙ���33?W
=@���CC�
�ٙ���p�?��HA<��C@B�                                    Bx�f�n  
�          @����(���?333@��\CB����(���(�?�ffA'�C@
                                    Bx�f�  �          @�p����Ϳ�=q?!G�@�Q�CB�{���Ϳ��R?�p�AffC@(�                                    Bx�f��  T          @�{���Ϳ��?(��@���CC8R���Ϳ\?��A%�C@k�                                    Bx�f�`  �          @����
��33?E�@�ffCCp����
��  ?�33A3�
C@O\                                    Bx�f�  �          @�{���H��(�?�G�A ��CD����H��p�?��AS
=C@=q                                    Bx�fѬ  �          @�  ��ff��p�?��
A\)CBz���ff��  ?�=qAPz�C>��                                    Bx�f�R  "          @�(���=q��(�?�=qA��CB����=q��(�?�\)AZffC>��                                    Bx�f��  "          @��
��=q���H?�ffAz�CB�)��=q��p�?˅AV=qC>�\                                    Bx�f��  T          @�(����H��33?��A=qCB
=���H��z�?�=qAT��C=�R                                    Bx�gD  �          @������H�޸R?���AffCBǮ���H���R?�\)AY�C>��                                    Bx�g�  "          @ۅ�ҏ\��Q�?xQ�A{CBc��ҏ\��p�?�G�AK�C>��                                    Bx�g)�  
�          @ۅ��G���  ?���A33CC  ��G����R?�33A_
=C>                                    Bx�g86            @ڏ\��G���z�?��AffCBB���G���?�=qAV�\C>(�                                    Bx�gF�  
          @�=q���ÿ�z�?��\A	CBG����ÿ�
=?�ffARffC>G�                                    Bx�gU�  T          @���У׿�
=?}p�A��CBn�У׿��H?��
AO\)C>�                                     Bx�gd(  
q          @ٙ��Ϯ��
=?��A33CB���Ϯ��
=?�\)A\z�C>E                                    Bx�gr�  
�          @�=q��  �У�?��RA((�CB
��  ����?�  Am�C=^�                                    Bx�g�t  �          @ڏ\��Q���?�p�A&�RCB#���Q쿊=q?޸RAl��C=s3                                    Bx�g�  
�          @���Ϯ��{?��
A,��CA��Ϯ���?�\Aq��C=
                                    Bx�g��  
�          @�����
=��33?��HA$��CBO\��
=����?�p�Al(�C=��                                    Bx�g�f  "          @����Ϯ����?�A�RCAٚ�Ϯ����?�Ad  C=W
                                    Bx�g�  �          @����θR��?�  AL��C@h��θR�G�?�A�G�C:�H                                    Bx�gʲ  T          @׮��=q��=q?�z�A��C?�
��=q�
=q@��A��RC8�f                                    Bx�g�X  �          @����θR����?�(�AH��C@�H�θR�O\)?�33A�(�C;+�                                    Bx�g��  �          @�\)��zῡG�?��HAk�C?&f��z���@33A�{C8��                                    Bx�g��  
�          @�z����H�p��?��HAn�RC<u����H��  ?�Q�A��C6L�                                    Bx�hJ  
�          @��H�ə���G�?У�Aep�C={�ə���{?��A��C7�                                    Bx�h�  �          @ҏ\��녿h��?���Aa�C<.��녾��?���A��\C6\)                                    Bx�h"�  �          @�G��ȣ׿�{?�(�AO�C>
=�ȣ׿   ?�\A{�C8�                                    Bx�h1<  T          @�����ÿ�?��AX��C>����ÿ�?�{A��C8�                                     Bx�h?�  T          @�33�ʏ\���?��HA(��C@W
�ʏ\�Y��?У�Aep�C;�)                                    Bx�hN�  "          @�(���z�\?O\)@߮CAc���z῏\)?��A6ffC=�                                    Bx�h].  "          @�����녿��R?�=qA��C?���녿B�\?��HAO33C:�{                                    Bx�hk�  �          @�ff���Ϳ�ff?�{A33CB&f���Ϳ��
?�{Af�HC=��                                    Bx�hzz  
�          @��ƸR���R?���Ap�C?O\�ƸR�@  ?�p�AT��C:��                                    Bx�h�   T          @�{��{��{?�{A�HC@aH��{�Y��?��
A\z�C;ٚ                                    Bx�h��  T          @�p������Q�?�\)AD��C>��������?��HAw33C9��                                    Bx�h�l  �          @�����׿c�
?��AqG�C<h����׾aG�?���A���C6�                                    Bx�h�  �          @�G����ÿ�G�?�  A\  C=� ���þ\?�G�A�(�C7�f                                    Bx�hø  "          @ʏ\��\)�k�?���A�C<���\)�8Q�@�A��C5                                    Bx�h�^  �          @ʏ\��33��  ?�\)AH  C=Y���33��
=?�33Ap��C7��                                    Bx�h�  
�          @ʏ\�Å��=q?��RA5p�C>��Å�
=q?ǮAc�C9�                                    Bx�h�  
�          @��H��G���=q?�z�AM�C@ff��G��5?�ffA��C:�3                                    Bx�h�P  T          @��
������  ?��
A:ffCA�f�����h��?�  A33C<��                                    Bx�i�  T          @˅��=q��  ?�\)A"ffCA�)��=q�xQ�?���Ah��C=)                                    Bx�i�  �          @����(���(�?�\)A!p�CAu���(��p��?˅AeC<�                                     Bx�i*B  �          @�z���{���?�A�C@u���{�   @G�A�z�C8ٚ                                    Bx�i8�  
�          @�33��Q��(�?   @�  CF#���Q���?���A.{CCE                                    Bx�iG�  �          @ʏ\��\)�G�>�@��HCF�3��\)�ٙ�?���A.�HCCٚ                                    Bx�iV4  
�          @�33��
=��?�ffAb�HCAk���
=�=p�?�(�A�33C;{                                    Bx�id�  "          @�p�������z�?�ffA�z�C>�\��������@
=A�
=C7�q                                    Bx�is�  �          @���G����?�G�A}C@\��G��
=q@
=A��C9�                                    Bx�i�&  �          @�{���׿��?��A�{C@{���׾��H@�RA�{C8�H                                    Bx�i��  T          @�����ÿ�?��Am��CAJ=���ÿ333@�
A�C:��                                    Bx�i�r  T          @�z���Q쿾�R?ǮAb�\CA޸��Q�J=q@ ��A�z�C;xR                                    Bx�i�  �          @��H��\)��\?��A{CD� ��\)���R?�33Aq��C?�                                     Bx�i��  �          @�����H��
=?��
A�CCn���H��?�=qAe�C>�H                                    Bx�i�d  �          @�
=�����p�?z�HA
�HCC�������p�?ǮA^�\C?O\                                    Bx�i�
  
�          @�ff��z��\)?�@�Q�CD�3��z���
?���A,  CB�                                    Bx�i�  "          @�����H��{?333@���CD����H����?���AB�\CAk�                                    Bx�i�V  T          @�{���
��G�?xQ�A33CD
���
���\?���Aa��C?��                                    Bx�j�  �          @�ff�Å��(�?�z�A%�CC�R�Å��33?�(�Aw\)C>�3                                    Bx�j�  
�          @У������\)?��A>{CB�3����}p�?���A�Q�C=�                                    Bx�j#H  T          @�(��љ���z�?Y��@��HCD=q�љ���Q�?�G�AJ�RC@n                                    Bx�j1�  T          @��
���ÿ��?n{@��CD����ÿ���?���ATQ�C?��                                    Bx�j@�  �          @�z����ÿ�G�?��A/33CC����ÿ���?��A}G�C=ٚ                                    Bx�jO:  "          @�����녿޸R?��RA%�CB�{��녿���?�As33C=�\                                    Bx�j]�  �          @���녿�p�?�z�A;\)CB��녿��?��HA�\)C=5�                                    Bx�jl�  
�          @�33�Ϯ���?�p�AG�
CB33�Ϯ�s33?��RA�\)C<^�                                    Bx�j{,  �          @�ff��=q��{?��
AS�CBJ=��=q�h��@�A�
=C<&f                                    Bx�j��  
�          @�p���녿У�?�33AA�CBxR��녿xQ�?�z�A��C<                                    Bx�j�x  
�          @�{���H��=q?���AE��CB�����H�p��?��A���C<                                    Bx�j�  
�          @����ÿУ�?��A^{CC)���ÿh��@33A�z�C<�H                                    Bx�j��  �          @���Q��
=?��A^ffCC�f��Q�u@z�A�Q�C=�                                    Bx�j�j  
�          @�{��  ���H?˅Ad��CC���  �z�H@�A�{C=:�                                    Bx�j�  
Z          @�Q���=q�ٙ�?У�Ah  CC���=q�s33@
=qA��HC<�f                                    Bx�j�  
�          @�p���\)��?���Af�RCC�)��\)�n{@�A�{C<�)                                    Bx�j�\  �          @�\)��녿�Q�?��A\��CC�=��녿xQ�@z�A�G�C=�                                    Bx�j�  �          @ٙ���(��޸R?ǮAT  CCJ=��(���G�@
=A��\C=�                                    Bx�k�  T          @�����Q��?�
=A8Q�CC5���Q쿓33@G�A��C=�f                                    Bx�kN  T          @�ff�ָR��z�?�\Ac�CC�H�ָR��=q@�A���C=+�                                    Bx�k*�  T          @������
?�(�A]�CE&f������R@Q�A���C>�                                    Bx�k9�  T          @�z���(���p�?�\Aep�CD�)��(���33@��A���C=��                                    Bx�kH@  "          @ᙚ��녿�\)?��Ak
=CC���녿��@�A��C=                                      Bx�kV�  �          @�\��=q��z�?�Am�CD0���=q���@=qA���C=0�                                    Bx�ke�  
�          @�\�љ���Q�?�{As33CD���љ���=q@p�A���C=T{                                    Bx�kt2  T          @�\�У׿�(�?�z�A{
=CD�{�У׿�=q@!�A�\)C=k�                                    Bx�k��  
�          @�\�����ff@	��A�\)CF0����Ϳ�{@2�\A�{C=ٚ                                    Bx�k�~  
�          @��H���p�@ ��A�  CF�����G�@-p�A�C?\                                    Bx�k�$  T          @�\��ff��@
=A�z�CEu���ff���@.�RA�33C=L�                                    Bx�k��  
�          @��H�љ�����?��Aw\)CD���љ�����@   A���C==q                                    Bx�k�p  
�          @��
���Ϳ�
=?��AUp�CD#����Ϳ��@��A�{C=�R                                    Bx�k�  T          @���ff���?�
=A9p�CC�H��ff��
=@�\A���C=�3                                    Bx�kڼ  "          @�z���=q��R?�AXz�CF����=q��33@��A�G�C@\                                    Bx�k�b  �          @�(���p���?�A8z�CEJ=��p���{@�A�Q�C?��                                    Bx�k�  
�          @�z������
=q?���A<  CF����Ϳ�
=@�A�(�C@)                                    Bx�l�  �          @�(���z��
=q?�
=A9CF
=��zῸQ�@
=qA�33C@33                                    Bx�lT  T          @�z������
�H?�33A4z�CF����Ϳ��H@Q�A��HC@\)                                    Bx�l#�  
�          @��
�����\)?�  A!CF�����Ϳ�=q@G�A��CAY�                                    Bx�l2�  
�          @�z���ff�(�?�33A  CF
��ff����?�z�Ax(�CA.                                    Bx�lAF  �          @��
������?���A�CF������?�Az=qCA�)                                    Bx�lO�  T          @�(���(��ff?�z�A�CG����(����H?�p�A�G�CBz�                                    Bx�l^�  �          @�z��ָR�(�?�ffA�CF
�ָR��{?���Al��CAxR                                    Bx�lm8  "          @���
=��?��A{CE8R��
=��G�?�\Af=qC@��                                    Bx�l{�  �          @ᙚ��p���\?u@��\CE  ��p���  ?�
=A\��C@�3                                    Bx�l��  "          @�G����
�	��?�G�AQ�CE�����
��=q?�\Ai�CAn                                    Bx�l�*  
�          @�����Q��   ?}p�A�CI  ��Q��z�?��Ax��CDaH                                    Bx�l��  
�          @�Q���33�z�?�{A�CEh���33��(�?�=qArffC@��                                    Bx�l�v  "          @�Q����H�p�?h��@�p�CF�����H��
=?ٙ�A`Q�CBW
                                    Bx�l�  T          @�Q���=q��?L��@��CG
��=q���
?У�AV�HCC&f                                    Bx�l��  o          @�  ��=q�\)?^�R@�CF�
��=q��(�?�
=A^�\CB�                                    Bx�l�h  k          @�Q���z���
?Q�@�  CEB���z��=q?���AN�RCAY�                                    Bx�l�  T          @�Q���(����?E�@ʏ\CE�f��(���?�ffALQ�CB)                                    Bx�l��  �          @�ff�����?Y��@ᙚCE�)��녿�{?�{AV�HCA��                                    Bx�mZ  �          @�\)���H�?u@�p�CE�{���H���?ٙ�Ab{CA0�                                    Bx�m   �          @�ff���
����?L��@ӅCDh����
��p�?�  AG�C@��                                    Bx�m+�  �          @�{���Ϳ�\)?:�H@���CC����Ϳ�
=?�33A:�\C@#�                                    Bx�m:L  "          @����Ӆ��33?#�
@���CD��Ӆ��  ?�=qA2=qC@�                                    Bx�mH�  
�          @ڏ\���H���H?�@��CB�=���H��\)?�33A
=C?�R                                    Bx�mW�  "          @ۅ����  >�G�@h��C@�������H?}p�A��C>:�                                    Bx�mf>  �          @�33����(�>�z�@=qC@k������R?W
=@�=qC>�                                    Bx�mt�  �          @�G���zῴz�>��@p�C@  ��zῘQ�?L��@�\)C>0�                                    Bx�m��  
�          @�=q��(��˅=u?�CAp���(���
=?.{@�\)C@0�                                    Bx�m�0  T          @ڏ\������R>8Q�?�G�C@�������ff?@  @ʏ\C?
=                                    Bx�m��  
�          @�=q��{���>�{@5C?\��{����?Tz�@�ffC=�                                    Bx�m�|  �          @�=q��{���>��@[�C>�H��{���\?aG�@�p�C<��                                    Bx�m�"  �          @�33��
=��  >��@|(�C>����
=�u?k�@�  C<!H                                    Bx�m��  �          @�33�ָR���>�p�@Dz�C?
�ָR���?\(�@�C=�                                    Bx�m�n  �          @أ���zῠ  >�{@:�HC>����zῂ�\?O\)@�z�C<�3                                    Bx�m�  �          @׮���Ϳ�{>�z�@p�C=z����Ϳh��?333@��C;                                    Bx�m��  "          @�\)���
��
=>�z�@�RC>!H���
�xQ�?:�H@ȣ�C<Y�                                    Bx�n`  
�          @�ff��=q���>�p�@J=qC?���=q���?Y��@���C=                                      Bx�n  
�          @��������  >���@4z�C>��������\?L��@ٙ�C<�R                                    Bx�n$�  �          @�Q���(�����>L��?ٙ�C?E��(�����?5@�  C=��                                    Bx�n3R  "          @������Ϳ��>\)?�(�C?^����Ϳ�?(��@��HC=�q                                    Bx�nA�  �          @�����(�����=�Q�?G�C?� ��(���p�?!G�@��HC>��                                    Bx�nP�  
�          @׮��녿�ff>�?���CAG���녿�\)?:�H@�Q�C?�=                                    Bx�n_D  
Z          @�Q���녿˅>W
=?޸RCA����녿���?Q�@�
=C?�f                                    Bx�nm�  T          @�{��  �˅>8Q�?��CA� ��  ���?J=q@�=qC@\                                    Bx�n|�  "          @�p���\)�Ǯ>k�?��HCA�\��\)���?Tz�@�z�C?�R                                    Bx�n�6  
�          @ҏ\�����p�>aG�?�z�CA�������\?J=q@�(�C?:�                                    Bx�n��  �          @�(����˅>�  @Q�CA�f����{?\(�@�C?�R                                    Bx�n��  
�          @�33���Ϳ�=q>.{?�  CA�H���Ϳ���?J=q@��HC@33                                    Bx�n�(  =          @�G���33�Ǯ>L��?�p�CA�
��33��{?L��@�G�C@�                                    Bx�n��  �          @������H���>u@z�CA�����H����?Tz�@���C?�q                                    Bx�n�t  �          @У��˅��
=>L��?�G�C@�3�˅���R?@  @�(�C?                                      Bx�n�  �          @�Q��˅��33>k�@ ��C@ff�˅��Q�?B�\@׮C>��                                    Bx�n��            @�Q����H��>�\)@   C@�{���H��Q�?Q�@��C>�
                                    Bx�o f  �          @����ə���Q�>�\)@�RCC\�ə���Q�?n{A\)C@��                                    Bx�o  �          @�����33��  >aG�?�z�CAE��33���?J=q@߮C?u�                                    Bx�o�  
Z          @�G����
���H>��@�\C@����
���R?Q�@�RC?�                                    Bx�o,X  
�          @�  ��=q��(�>aG�?�z�CA!H��=q���\?G�@�{C?W
                                    Bx�o:�  �          @�p���Q쿴z�=�?�{C@����Q쿠  ?+�@���C?B�                                    Bx�oI�  �          @�(��ƸR����>��R@1G�CA��ƸR����?\(�@�\)C>�q                                    Bx�oXJ  
�          @����Ǯ��33>u@	��C@���Ǯ��Q�?G�@߮C>�{                                    Bx�of�  4          @ʏ\�����z�>���@0��C@�)�����?W
=@��
C>�                                     Bx�ou�  �          @�=q��녿�
=?
=@��
CC����녿��?��HA1�C@5�                                    Bx�o�<  �          @�G���=q�\?   @��CB
��=q����?��A  C?+�                                    Bx�o��  �          @ə���=q�Ǯ>�33@L��CBff��=q���?p��A
{C@                                      Bx�o��  �          @�=q���׿�z�?G�@��
CCaH���׿��H?�\)AI�C?\)                                    Bx�o�.  �          @��
��G���
=?�=qA(�CC���G���{?�z�Aq��C>h�                                    Bx�o��  �          @�
=�ƸR��ff?uA(�CB
=�ƸR���?�  AV{C=�=                                    Bx�o�z  �          @�{������{?�
=A(z�CE�������p�?���A�G�C?�                                    Bx�o�   �          @˅��33����?�As�CE����33���
@G�A��C=�q                                    Bx�o��  T          @�33������\)?��
A�{CE�
������G�@�A��C=�{                                    Bx�o�l  �          @�z���p���\)?��
A]p�CE����p���{@	��A��C>��                                    Bx�p  �          @�p�����ff?��AG\)CG�������\)@
=A�33CA�                                    Bx�p�  �          @�(������?�(�A0z�CD�\�����?�A�(�C?{                                    Bx�p%^  �          @�(���Q��ff?���A#�
CD���Q쿙��?�G�A�  C?J=                                    Bx�p4  �          @�(���G���\?�G�A�\CDQ���G���(�?��Am�C?c�                                    Bx�pB�  �          @��
�����\)?�{A ��CEW
������\?�\A���C?�q                                    Bx�pQP  �          @��
��G���=q?\(�@���CD�f��G����?\A]G�C@�                                     Bx�p_�  �          @��
������G�?uA
�\CD5�������p�?�=qAe�C?}q                                    Bx�pn�  �          @ʏ\��  ���
?s33A
�RCD� ��  ��  ?�=qAg�
C?�                                    Bx�p}B  �          @ə���\)���
?W
=@�ffCD����\)��ff?�p�AZ=qC@E                                    Bx�p��  �          @����(���Q�?J=q@�CD���(����R?�33AQ�C?��                                    Bx�p��  T          @�ff��z���H?fffAp�CD:���zῚ�H?�G�AaG�C?��                                    Bx�p�4  �          @�p���녿޸R?��A)CD���녿��?޸RA��RC?#�                                    Bx�p��  �          @�(����ÿ�  ?���A (�CD�{���ÿ�
=?�A}��C?��                                    Bx�pƀ  �          @�z���ff��?���AG
=CE޸��ff��z�?���A��
C?}q                                    Bx�p�&  �          @��H���
��\)?�\)AP(�CFaH���
��@   A��C?��                                    Bx�p��  �          @��H��
=�G�?޸RA��HCH@ ��
=��z�@��A��
C?�3                                    Bx�p�r  �          @�=q��\)��(�?ٙ�A�z�CG��\)����@ffA�(�C?�)                                    Bx�q  �          @�G����ÿ��
?�33A}p�CE�)���ÿz�H@{A��RC>                                    Bx�q�  �          @\�����z�?��HA^�RCDc�����p��@   A�{C=u�                                    Bx�qd  �          @������Ϳ˅?��AN�\CC�����Ϳk�?�\)A��\C=G�                                    Bx�q-
  �          @��
�����=q?�A��CDff����5@z�A���C;��                                    Bx�q;�  �          @�z���(����?�A�p�CD���(��.{@33A��C;:�                                    Bx�qJV  �          @����(����@�A���CBu���(���@�HA��C8��                                    Bx�qX�  �          @�{���׿�Q�?�
=A��CB�
���׿&ff@
=A�ffC:��                                    Bx�qg�  �          @�  ��33��p�?��AnffCB�\��33�=p�@   A�=qC;�\                                    Bx�qvH  �          @�Q����
��ff?�p�Ad��CCu����
�Tz�?�p�A��
C<n                                    Bx�q��  �          @�������  ?��HAb=qCC  ����L��?�Q�A�G�C<�                                    Bx�q��  �          @�  ��p��˅?�33A0(�CC���p���  ?�
=A��\C>�                                    Bx�q�:  �          @�  ���Ϳ˅?�  A@  CC� ���Ϳu?��
A�  C=�                                    Bx�q��  �          @�����33��Q�?�
=A�p�CBh���33�&ff@
=A�Q�C:�H                                    Bx�q��  �          @�����녿�{?���A�p�CAǮ��녿�\@\)A�
=C9B�                                    Bx�q�,  �          @�{��{����@   A�\)CA�3��{��
=@
=A�p�C8ff                                    Bx�q��  �          @�{���ÿ˅?�A^=qCD\���ÿc�
?�Q�A���C=+�                                    Bx�q�x  �          @��R���\�G�>���@w
=CG�f���\��Q�?�A5CD�H                                    Bx�q�  �          @��R��(���=q?#�
@�{CF��(���?���AL��CB0�                                    Bx�r�  �          @����
=� ��?
=q@���CH33��
=�У�?�ffAL��CD�)                                    Bx�rj  �          @��R������z�?��A ��CF�R�������?��HA���CA�=                                    Bx�r&  �          @\��z���
?�{AO�CE����zΉ�?��HA���C>�R                                    Bx�r4�  �          @��
����޸R?�\)Ak33CD\)����s33@
�HA���C=)                                    Bx�rC\  �          @����=q��z�?��A��CC޸��=q�O\)@33A���C;��                                    Bx�rR  �          @������\��
=?У�ApQ�CD\���\�c�
@
=qA�
=C<�3                                    Bx�r`�  �          @���������G�?�33As�CD޸�����u@p�A���C=^�                                    Bx�roN  �          @�����G��ٙ�?��HA}G�CDaH��G��aG�@  A��C<�f                                    Bx�r}�  �          @�����녿�
=?�Q�Az{CD#���녿^�R@{A��C<��                                    Bx�r��  �          @�Q����׿�  ?�Q�A{33CDٚ���׿n{@  A�ffC=&f                                    Bx�r�@  �          @�
=����ff?�ffA�=qCE�����p��@�A��RC=ff                                    Bx�r��  �          @ȣ���Q���
?�(�A33CE.��Q�s33@�\A�G�C=\)                                    Bx�r��  �          @�  ��  ���?��A��CC����  �J=q@33A��RC;�\                                    Bx�r�2  �          @�  ��\)�ٙ�?���A�G�CD����\)�W
=@A�ffC<\)                                    Bx�r��  �          @�\)��Q���?�
=A{33CC����Q�W
=@��A�p�C<G�                                    Bx�r�~  �          @Ǯ���H��ff?\AaG�CB�)���H�Q�@ ��A�Q�C<                                      Bx�r�$  �          @�{���
���H?�G�AC�CB�����
�W
=?޸RA��C<�\                                    Bx�s�  �          @�p���
=�ٙ�?��
Af=qCD����
=�s33@�A��
C=s3                                    Bx�sp  �          @�
=���ÿ��?˅AmCC�
���ÿ^�R@
=A�
=C<�
                                    Bx�s  �          @�G���33���?���AqG�CDQ���33�aG�@A�(�C<�3                                    Bx�s-�  �          @�{����У�?�z�Ax  CC�\����W
=@
=qA��C<J=                                    Bx�s<b  �          @�
=��\)��\)?�p�A�z�CDs3��\)�L��@�RA�33C<Q�                                    Bx�sK  �          @�(���\)��  ?�
=Alz�CC����\)�O\)?�z�A��HC<�                                    Bx�sY�  �          @����Q�˅?�33A5��CD!H��Q쿁G�?�Q�A�=qC>\)                                    Bx�shT  �          @Ǯ��=q���?\Aa��CCǮ��=q�h��@33A�33C<޸                                    Bx�sv�  �          @��
���޸R?��
A^=qCDW
���}p�@A�p�C=}q                                    Bx�s��  T          @�{���ÿ�
=?���AXQ�CD33���ÿxQ�?��RA�=qC=�                                    Bx�s�F  �          @����׿�33?�
=AW
=CC�����׿s33?�(�A���C=Y�                                    Bx�s��  �          @�33����  ?�Q�AQp�CDn�����@ ��A��
C=�                                    Bx�s��  �          @��H��p���  ?�Q�AQCD����p����@ ��A�=qC>                                      Bx�s�8  �          @�p����ÿ�p�?�\)AECC�R���ÿ��?�Q�A�
=C=��                                    Bx�s��  �          @�z����R��ff?��AI�CD�����R����?��RA���C>�                                     Bx�s݄  �          @�Q�������H?�ffAG�
CD�f�����ff?�\)A��HC>�H                                    Bx�s�*  �          @�33�������?��HA?33CD@ ����}p�?�  A��HC>@                                     Bx�s��  �          @�\)���ÿ�p�?�Q�A^�\CE^����ÿ��\@   A��RC>}q                                    Bx�t	v  �          @Å�����\)?�ffAD��CC�R����xQ�?�=qA��C=�)                                    Bx�t  �          @�
=��  ����?��
AdQ�CE����  ����@Q�A�{C>��                                    Bx�t&�  �          @�=q��(����
?�Q�A\  CE����(�����@�A�{C>�                                     Bx�t5h  
�          @�����Ϳ�Q�?��AV�\CBL����ͿG�?�A�p�C;��                                    Bx�tD  �          @��R�����=q?�ffAJ=qCA33����5?�(�A�Q�C;�                                    Bx�tR�  �          @�ff���Ϳ���?��\AD��CAu����Ϳ@  ?ٙ�A��\C;�                                    Bx�taZ  �          @��
���ÿ���?��ANffCB�����ÿQ�?�\A���C<c�                                    Bx�tp   �          @����
=��(�?��
AK�
CC���
=�Y��?�G�A���C<޸                                    Bx�t~�  T          @�p����H����?��RAIp�CC.���H�Y��?��HA�{C=�                                    Bx�t�L  �          @�z���\)�޸R?�
=A@��CF\)��\)����?�G�A��C@:�                                    Bx�t��  �          @�  ���Ϳ�ff?�(�ADz�CD����Ϳp��?�p�A�Q�C=�                                    Bx�t��  �          @��\������H?��HA?�
CH��������?��A���CA��                                    Bx�t�>  �          @�G���=q��Q�?�Q�A=��CG�q��=q���?���A�\)CA�H                                    Bx�t��  �          @�������p�?��\AL  CC5����\(�?�  A�G�C<�q                                    Bx�t֊  �          @�\)����\)?��RAH(�CB0����G�?�
=A�Q�C<!H                                    Bx�t�0  T          @�p���G���=q?�ffAS�
CD����G��p��?�A��C>�                                    Bx�t��  �          @�
=�����?�33AbffCG^������33@ ��A�Q�C@W
                                    Bx�u|  �          @��H��33��?�Aa�CF����33���@�A�z�C@                                      Bx�u"  �          @������H��Q�?�33A`��CE�
���H��G�?���A�  C>��                                    Bx�u�  �          @����p��Ǯ?�\)A[33CD���p��c�
?�\)A��\C=^�                                    Bx�u.n  �          @�������У�?�z�Aa�CD�����s33?�Q�A��\C>{                                    Bx�u=  �          @�p����R��G�?���AbffCE�\���R��ff@G�A�p�C>��                                    Bx�uK�  �          @������
�H?�Q�A`��CJ33��녿�Q�@
�HA�  CC!H                                    Bx�uZ`  �          @��H����� ��?�\)AX��CH���������@�\A��CA�R                                    Bx�ui  �          @�p�������?���A[�
CH������(�?�p�A�p�CAG�                                    Bx�uw�  �          @�����
=��p�?�ffAT��CFO\��
=��=q?�\)A�z�C?�q                                    Bx�u�R  �          @�����(���\?�(�AC33CF5���(���33?���A�=qC@)                                    Bx�u��  �          @�=q���Ϳ�?�
=A:�HCF�{���Ϳ��R?�ffA�=qC@��                                    Bx�u��  T          @�z����
����?���AXz�CHٚ���
���
?�(�A��CB{                                    Bx�u�D  �          @�  ��
=� ��?��AX  CI���
=��=q@G�A�=qCBJ=                                    Bx�u��  �          @������R� ��?��
At��CI!H���R��G�@(�A��
CA�)                                    Bx�uϐ  �          @�p����Ϳ�z�?�z�Af�\CHO\���Ϳ��H@�\A��RCA8R                                    Bx�u�6  �          @�����\)���
?�z�A=p�CF�\��\)��Q�?�G�A�33C@ٚ                                    Bx�u��  �          @�Q���=q��\)?�Q�A?
=CG^���=q��G�?���A��CAY�                                    Bx�u��  T          @�����׿��?��RAHz�CG�����׿�  ?��A��RCA^�                                    Bx�v
(  �          @�����\��?�z�A`��CF�3���\��\)@   A���C?�)                                    Bx�v�  �          @��������
?�\)AZffCFc������{?���A��C?��                                    Bx�v't  �          @�(���{��=q?��AM��CF�)��{��
=?�z�A��
C@G�                                    Bx�v6  �          @�p���
=��\?�33A[\)CE����
=���?�p�A�=qC?O\                                    Bx�vD�  �          @�=q��(���  ?���AZ�HCF���(���=q?�Q�A�(�C?ff                                    Bx�vSf  �          @�
=��=q��p�?���AAp�CE����=q����?�\A�Q�C@�                                    Bx�vb  �          @�������Q�?�z�A:{CExR�����{?�(�A�\)C?�R                                    Bx�vp�  �          @����z���H?��APz�CE�
��zῈ��?�\)A�  C?B�                                    Bx�vX  �          @����
=��(�?�
=A:{CEh���
=����?�  A��C?��                                    Bx�v��  �          @�G���ff�˅?���A0  CDE��ff���?У�A���C>ٚ                                    Bx�v��  �          @�33��G���33?k�A(�CD�
��G���?�p�AjffC?�                                    Bx�v�J  �          @�����ÿ�{?:�H@�  CD0����ÿ���?��AM�C@B�                                    Bx�v��  �          @�����=q��?#�
@�{CE����=q��
=?��
AH��CB\)                                    Bx�vȖ  �          @�
=����
=?@  @�ffCDs3����G�?�=qAN�\C@��                                    Bx�v�<  �          @�G����׿�?��@��RCD+����׿���?���A6�HC@ٚ                                    Bx�v��  �          @Å��=q��  ?\)@�\)CD�3��=q��33?�
=A2�RCA��                                    Bx�v�  �          @�z���  ���?�A��CA33��  ���@�A���C9                                    Bx�w.  �          @�(���Q쿞�R?У�A�
=C@����Q��?�p�A�C9p�                                    Bx�w�  �          @�{��G����?�(�A�ffCA#���G��
=q@�A�  C9�\                                    Bx�w z  T          @�ff��G����?�(�A���CAG���G����@A���C9��                                    Bx�w/   �          @�G���(���=q?�(�A�z�CAG���(���@ffA���C9��                                    Bx�w=�  T          @��R���H���
?���Ax��C@����H�z�?�(�A�=qC9��                                    Bx�wLl  �          @�  ��zῪ=q?ǮAp��CAO\��z�#�
?���A��C:}q                                    Bx�w[  �          @�Q�������?ٙ�A��CAh�����
=@�A�=qC:                                      Bx�wi�  �          @����Q쿫�?У�A�z�CA����Q��R@G�A�{C:h�                                    Bx�wx^  �          @����z῱�?�A�Q�CBs3��z�
=@�RA��RC:8R                                    Bx�w�  �          @����\)��\)?�  A��CA�q��\)���@��A��C:B�                                    Bx�w��  �          @�  ��33����?�A��CA�f��33�&ff@z�A��C:�)                                    Bx�w�P  �          @\��p���33?�Q�A��CA�f��p��(��@ffA�G�C:��                                    Bx�w��  �          @�\)�������?�=qAuCA�{����(��?�p�A�z�C:��                                    Bx�w��  T          @�{���׿��?�  A�33CAc����׿\)@�A��C9�                                     Bx�w�B  
�          @��R���R��\)?�(�A�ffCB����R��@A��C9��                                    Bx�w��  T          @�
=��=q��G�?�  A�z�C@���=q��\@ffA��\C98R                                    Bx�w�  �          @�p����\��?˅Ayp�C?�\���\��?�z�A��HC8�                                    Bx�w�4  �          @�=q���ÿ�z�?�AX(�C?aH���ÿ��?�  A�=qC9c�                                    Bx�x
�  �          @�����׿�z�?���A]G�C?aH���׿�?��
A��\C9E                                    Bx�x�  �          @�Q���p���z�?�{Ax��C?����p����?�
=A��C8Ǯ                                    Bx�x(&  �          @�������
=?�{Aw�
C?� �����H?�
=A��C8�3                                    Bx�x6�  �          @��H��녿�\)?��A]p�C?����녿
=q?�z�A�{C9                                    Bx�xEr  �          @�(���33����?�33Aep�C?L���33���?�Q�A�{C9�                                    Bx�xT  �          @�\)��{���\?�  Ar{C>����{�Ǯ?�\A��C8)                                    Bx�xb�  �          @�\)���}p�?�G�Aup�C>^�����Q�?��
A��\C7                                    Bx�xqd  �          @�����׿n{?�  Az=qC>
=���׾��R?޸RA��
C7T{                                    Bx�x�
  �          @�����H�p��?���Ao33C=����H����?�Q�A�ffC7��                                    Bx�x��  �          @�=q��=q���?�p�AB�RC?�)��=q�(�?ǮAy��C:33                                    Bx�x�V  �          @��
��(����?��RAC
=C>����(����?�ffAuC9��                                    Bx�x��  �          @�����(��k�?�  Ak33C=@ ��(�����?�p�A�z�C7                                      Bx�x��  �          @�ff��ff�k�?�A\��C=#���ff���
?�z�A���C7B�                                    Bx�x�H  �          @�\)��\)����?��AG�C>����\)��?���Aw�
C9!H                                    Bx�x��  �          @�=q���H��33?�Q�A4(�C?+����H�!G�?��
AiG�C:.                                    Bx�x�  �          @�=q���Ϳ�\)?J=q@�RC>�����Ϳ@  ?�33A.�\C;J=                                    Bx�x�:  �          @������׿k�?�(�A<��C=
=���׾���?�(�Ad��C7�q                                    Bx�y�  �          @��
���Ϳ��@33A�(�C9�\����>�{@A��C0�
                                    Bx�y�  �          @�������@z�A��C8����>��@�A�=qC/�H                                    Bx�y!,  �          @�������333@G�A�(�C;����=���@��A�\)C3�                                    Bx�y/�  T          @��H���R�8Q�@   A���C;#����R=�\)@�A��HC3O\                                    Bx�y>x  �          @��H��p��+�@�A��HC:��p�>��@{A��HC2u�                                    Bx�yM  �          @���������@�A�33C:{���>B�\@
=A���C2�                                    Bx�y[�  �          @�\)���Ϳ�R?�{A���C:8R����=�G�?��HA��RC2�f                                    Bx�yjj  �          @�  ��p��(�?��A��HC:���p�>�?�p�A�=qC2��                                    Bx�yy  �          @�33��
=��@33A�
=C9����
=>aG�@�A�Q�C1�\                                    Bx�y��  �          @��H���z�@�A��C9޸��>k�@(�A�z�C1��                                    Bx�y�\  �          @������5?�A��C;
��=#�
@�\A�  C3�
                                    Bx�y�  �          @\��
=�333?�
=A��
C;  ��
==L��@33A�C3��                                    Bx�y��  �          @ȣ���p��@  ?�
=A�C;8R��p�<#�
@z�A��\C3�                                    Bx�y�N  �          @�=q���aG�?��RA��C<ff����Q�@
�HA��C4��                                    Bx�y��  �          @����녾�(�?�(�A��C8:����>��
@   A�\)C0��                                    Bx�yߚ  �          @�G���{����@�A���C7����{>�Q�@�\A�p�C0��                                    Bx�y�@  �          @��H��\)��(�@�
A��C8#���\)>�{@�A���C0�q                                    Bx�y��  �          @�G���ff���H?�p�A���C8�3��ff>��@G�A��C1��                                    Bx�z�  �          @�\)�����?��A��RC9���>B�\?�Q�A�p�C2)                                    Bx�z2  �          @�����녿��?�33A�(�C9����=�?��RA�33C2�=                                    Bx�z(�  �          @����\��?��RA��RC8�����\>�z�@�A�33C1(�                                    Bx�z7~  
�          @�  ��(���\@�
A�  C9  ��(�>��@
=A��C1u�                                    Bx�zF$  �          @˅��
=�\)@�A��C9O\��
=>k�@��A�C1��                                    Bx�zT�  �          @�Q����R�G�?�p�A�Q�C;z����R��G�?��A�
=C5\                                    Bx�zcp  �          @�G����ÿ�\)?���AN�\C>�q���ÿ\)?�z�A�C9�{                                    Bx�zr  �          @����  ��{?��
AEp�C>�3��  �z�?˅Av�RC9                                    Bx�z��  �          @\��33���?��AC�C>G���33��?�=qAqp�C9.                                    Bx�z�b  �          @��H�����ff?��AC
=C>)�����?���Ao�C9�                                    Bx�z�  T          @��H��(����?�p�A9�C>���(��
=q?\Af�RC95�                                    Bx�z��  �          @�������  ?�A1�C=�R�����?���A]p�C9�                                    Bx�z�T  �          @\��(���=q?�\)A*ffC>n��(��(�?�
=AZffC9�3                                    Bx�z��  �          @�z������\)?��HA5G�C>�3�����R?\Ae�C9��                                    Bx�zؠ  �          @�������G�?�z�A0��C=�=�����?���A\��C95�                                    Bx�z�F  �          @Å����  ?�ffA
=C=�\���\)?��AJ�RC9h�                                    Bx�z��            @��H��z�Tz�?��AC33C<���zᾣ�
?�  Ac�
C7#�                                    Bx�{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{?*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{\v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{k              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{Ѧ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�{��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�|>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�|)�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�|80              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�|F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�|U|  �          @Å��33�8Q�����RC5����33�8Q�<�>�z�C5��                                    Bx�|d"  �          @�z���(��u�k����C6E��(����R����  C6�)                                    Bx�|r�  �          @�����G�����L�Ϳ�z�C6s3��G����R���ͿxQ�C6�3                                    Bx�|�n  �          @�  �������.{�˅C6s3���������\)�(��C6ٚ                                    Bx�|�  �          @�����Q쾏\)��z��333C6����Q쾸Q�8Q�޸RC7p�                                    Bx�|��  T          @\�����L�Ϳ\)����C5�f�����\�����\C7��                                    Bx�|�`  o          @��
�ʏ\�����.{�\C6�R�ʏ\��\�����C8�)                                    Bx�|�  �          @��
�ʏ\����:�H���HC5T{�ʏ\�\�#�
��Q�C7xR                                    Bx�|ʬ  �          @�=q��Q�L�ͿTz���=qC4xR��Q쾨�ÿE���
=C7                                      Bx�|�R  �          @�G���
==��Ϳn{��C3���
=�aG��h���z�C6�                                    Bx�|��  �          @����ƸR=#�
�u�(�C3���ƸR��z�h���C6��                                    Bx�|��  �          @�G���ff=�G�����  C3��ff�������(�C6\)                                    Bx�}D  �          @�G���{>����=q��C2�H��{�aG�����  C6�                                    Bx�}�  �          @ə���ff�\)�����HC5B���ff���s33�33C8n                                    Bx�}"�  �          @Ǯ��zᾨ�ÿ��
��
C7���z�&ff�\(���C:\                                    Bx�}16  �          @������þ�����p�C8^����ÿO\)�^�R���HC;c�                                    Bx�}?�  �          @�\)��33�0�׿�\)��RC::���33���\�Q����C=(�                                    Bx�}N�  �          @θR���H�0�׿�ff�G�C::����H��  �B�\��  C<��                                    Bx�}](  �          @���G��E�������C:����G���=q�B�\�ٙ�C=�q                                    Bx�}k�            @Ϯ��33�:�H�����(�C:���33��ff�L����\C=\)                                    Bx�}zt  �          @�Q����
�TzῊ=q��C;ff���
��녿@  ���
C>�                                    Bx�}�  �          @љ����ͿY����{�z�C;�\���Ϳ��E�����C>W
                                    Bx�}��  �          @��H��{�Y����=q��C;�=��{��z�=p��ϮC>5�                                    Bx�}�f  �          @��H��ff�W
=��=q��C;p���ff��33�@  ����C>)                                    Bx�}�  �          @ҏ\��{�W
=���
�(�C;s3��{���׿333�ÅC=�R                                    Bx�}ò  �          @�����p��#�
��G���\C9�3��p��p�׿@  �ҏ\C<J=                                    Bx�}�X  �          @��
��Q�#�
��G��  C9����Q�n{�=p���ffC<+�                                    Bx�}��  �          @Ӆ��\)�:�H����p�C:p���\)���
�@  �У�C=�                                    Bx�}�  �          @�(���  �=p��������C:����  ��ff�E��ָRC=0�                                    Bx�}�J  �          @љ�����@  ������C:��������J=q����C=ff                                    Bx�~�  �          @�G����ͿJ=q�����HC;����Ϳ�=q�:�H��C=��                                    Bx�~�  �          @ҏ\���h�ÿ��
�(�C<�����Q�.{��
=C>z�                                    Bx�~*<  T          @��
��
=�}p��}p����C<�R��
=��G��(����\C>�q                                    Bx�~8�  �          @�(���\)���\�p���33C<���\)���\����C?�                                    Bx�~G�  �          @�z���  ��  �p���=qC<���  ��  ����C>��                                    Bx�~V.  �          @�z��Ϯ���
�u���C<�q�Ϯ���
�z���Q�C?#�                                    Bx�~d�  �          @����Q쿂�\�z�H�  C<�f��Q쿣�
�(���  C?!H                                    Bx�~sz  �          @����  ���
��G��  C<����  ����#�
��\)C?B�                                    Bx�~�   �          @��
��\)�n{�}p��	C<+���\)�����#�
���HC>u�                                    Bx�~��  �          @��
��\)�}p��s33��C<����\)���R�
=��=qC>�
                                    Bx�~�l  �          @�=q�θR�s33�J=q��{C<\)�θR��33��ff�}p�C>)                                    Bx�~�  �          @�=q��{�fff�c�
��Q�C;����{��녿\)���C>                                      Bx�~��  �          @љ���{�\(��Y����RC;�)��{��=q�
=q��ffC=�\                                    Bx�~�^  �          @�=q��
=�J=q�G����HC:���
=�}p��   ���\C<��                                    Bx�~�  �          @�33��  �@  �Tz���C:����  �z�H������C<�=                                    Bx�~�  �          @�p��љ��\(��n{���C;� �љ���{�(���Q�C=�)                                    Bx�~�P  �          @�ff��G���Q�\(���C>T{��G���33��G��s33C@
                                    Bx��  �          @�����\)���׿n{� z�C=�f��\)��{�����\C?��                                    Bx��  T          @����Ϯ�\(�����{C;���Ϯ���Q���=qC>5�                                    Bx�#B  �          @�(���
=�\(����"�RC;����
=��
=�Y�����HC>W
                                    Bx�1�  �          @�=q���Ϳc�
��z��"ffC;����Ϳ��H�Q���RC>�                                    Bx�@�  �          @�33��{�W
=��z��!p�C;k���{��33�W
=��=qC>(�                                    Bx�O4  �          @�=q����fff��33�!C;��������H�Q���C>��                                    Bx�]�  �          @У����
�L�Ϳ�\)�ffC;+����
��{�Q���\)C=ٚ                                    Bx�l�  �          @�z���\)�@  ��Q��+�C:����\)��=q�fff�C=�
                                    Bx�{&  �          @�z��Ǯ�B�\����$Q�C:�q�Ǯ��=q�Y������C=�=                                    Bx���  �          @��H��{�=p���z��((�C:ٚ��{����^�R��C=��                                    Bx��r  �          @˅��\)�333������C:p���\)��  �Q���C=!H                                    Bx��  �          @�(����ÿ&ff�n{��\C9�����ÿfff�0����\)C<(�                                    Bx���  �          @�33��  �8Q�aG���p�C:�{��  �s33��R��=qC<��                                    Bx��d  �          @��
��Q�G��Y�����RC;���Q쿀  �z���
=C=\                                    Bx��
  �          @�(���G��@  �G��߮C:����G��s33��\���
C<�{                                    Bx��  �          @������ÿO\)�c�
� z�C;Y����ÿ���(���ffC=k�                                    Bx��V  �          @�\)��(���\)��\�~ffC>Y���(����Ϳ�{�Ap�CB�H                                    Bx���  �          @Ϯ���
��녿�{���HC>�\���
��녿�
=�K�CC
=                                    Bx���  �          @�
=���H��(��������HC?J=���H�ٙ������Dz�CC��                                    Bx��H  �          @����  ��33�����C>����  �����R�W�CC��                                    Bx��*�  �          @�33��33��  �
=��
=C@\��33���ÿ���o�CEE                                    Bx��9�  �          @�(���33��z��
=���RCA����33��(������hQ�CF��                                    Bx��H:  �          @�{��ff��������C@� ��ff��׿�{�g33CE��                                    Bx��V�  �          @����������
�H���HC@���������
=�s�CF(�                                    Bx��e�  �          @��
��(���=q��\��C@�q��(���\)�Ǯ�c
=CE��                                    Bx��t,  �          @˅��z῱녿��H��G�CAL���z��zῺ�H�Tz�CE�                                    Bx����  �          @�ff��녿���{��  C>�H��녿�zῸQ��MCCL�                                    Bx���x  �          @�p���  ���
�������C@���  ��\��
=�N{CDz�                                    Bx���  �          @��H���R��녿���33C>�\���R�У׿�
=�P��CCB�                                    Bx����  �          @��H����
=�������C?E����
=���H�Up�CC�\                                    Bx���j  �          @��
����z���R��{C?)����Q�����d��CC�                                    Bx���  �          @˅���R��{��33��G�C>�\���R��{��  �Z=qCC&f                                    Bx��ڶ  �          @ʏ\��z῕�G����RC?33��z�ٙ������ip�CD�                                    Bx���\  �          @��
��(���\)�
�H���C>����(��ٙ���  �\)CD!H                                    Bx���  �          @��H���H��33�
�H��  C?����H��(���  ��
CDn                                    Bx���  �          @����G���{�ff���CA33��G���33�У��o
=CF5�                                    Bx��N  �          @�����  ��33�ff��=qCA�f��  ��Q��{�n{CF��                                    Bx��#�  �          @ȣ����\��
=��p���\)C?}q���\�ٙ��Ǯ�f�\CDB�                                    Bx��2�  �          @ə���=q��z��ff���C?G���=q���H��
=�w\)CD\)                                    Bx��A@  �          @�����=q��������C>L���=q��{�ٙ��|  CCp�                                    Bx��O�  �          @ʏ\��33�����Q�����C>����33��Q��p��}p�CD�                                    Bx��^�  �          @ʏ\�������Q���\)C>}q�����녿޸R�~�HCC�f                                    Bx��m2  �          @�  ���ÿs33�
=q���C=T{���ÿ\��ff��{CB��                                    Bx��{�  �          @�  ��Q�p���  ���\C=:���Q���
������CB޸                                    Bx���~  �          @�Q����ÿ��H�z����HC?�����ÿ޸R��33�t��CDǮ                                    Bx���$  �          @�G���=q����z����CA����=q��녿�
=�R�HCF�                                    Bx����  �          @ƸR��ff��(��
=��33C@���ff��G���Q��|��CE#�                                    Bx���p  �          @����=q���R��
����C@�\��=q��=q�����{CF:�                                    Bx���  �          @�  ��\)��  �	�����RC@Q���\)��ff���H�~�HCEh�                                    Bx��Ӽ  �          @�ff���R����   ���\CA(����R����ff�h  CE��                                    Bx���b  �          @�ff��{��(��	����  C@)��{��G���p���\)CE=q                                    Bx���  T          @�{���Ϳ��\�33����C>G����Ϳ�\)��
=��p�CC��                                    Bx����  �          @������(������HC:�������
�z���33C@�q                                    Bx��T  �          @������
�E��
=���\C;� ���
������
����CA��                                    Bx���  �          @�{��p��Q�������C<5���p����G���Q�CB{                                    Bx��+�  �          @�{��p��aG���
���C<Ǯ��p���(����R��CB�=                                    Bx��:F  �          @ə���33�Y���
�H��33C<@ ��33��33��{��G�CA�                                     Bx��H�  �          @ȣ�����J=q���C;�3�����=q��ff��G�C@�                                    Bx��W�  �          @����ff�0�׿��R����C:�H��ff������p��}C?h�                                    Bx��f8  �          @˅��Q�@  �����RC;���Q쿞�R��33�pz�C?��                                    Bx��t�  �          @��
��  �n{��
=����C<���  ��z��{�i�CA5�                                    Bx����  T          @�(����ÿk���{���C<����ÿ�녿�ff�`Q�C@�                                    Bx���*  �          @�p����H�aG��������C<B����H�������^�\C@s3                                    Bx����  �          @�=q��Q�O\)��  ���\C;����Q쿞�R��(��W�C?��                                    Bx���v  �          @�����  �G���z��v{C;aH��  ��Q쿳33�N{C?8R                                    Bx���  �          @˅��33�E���\)�lz�C;8R��33����{�E��C>��                                    Bx����  �          @�����G��W
=��p��Z{C;����G��������H�1G�C?5�                                    Bx���h  �          @˅���
�n{��(��V=qC<�H���
���
��
=�*�\C?ٚ                                    Bx���  �          @У���  �����p��Q��C=�H��  ��Q쿑��!�C@�3                                    Bx����  �          @�����ÿ�����Q��K�C>����ÿ��
��=q��
CA�3                                    Bx��Z  �          @У������{��\)�f�RC@p�����޸R���H�+�CC��                                    Bx��   �          @�Q����H��������
=C@����H��G������L  CD                                    Bx��$�  �          @���Q쿳33�޸R���\CA����Q��ff����Dz�CEaH                                    Bx��3L  �          @�ff���ÿ�
=��(���
=CA޸���ÿ��ÿ���@��CE��                                    Bx��A�  �          @�z���\)���ٙ���z�CA����\)�����
�@  CE�                                     Bx��P�  �          @�(������=q���}p�CA  ������H���
�@Q�CD�
                                    Bx��_>  �          @�G��������=q�qCAB����ٙ���
=�4(�CD��                                    Bx��m�  �          @�����p���녿�  �f�\CA�q��p���(������'�CD��                                    Bx��|�  �          @�G���33���H�޸R���CB����33��{����Hz�CFY�                                    Bx���0  �          @����=q��G�����G�CC!H��=q������T��CG�                                    Bx����  �          @�Q�������
��p��d(�CCB������{���� (�CFJ=                                    Bx���|  �          @��R����˅��(��=G�CC�{������ͿG�����CF:�                                    Bx���"  �          @ҏ\��z��  �Ǯ�\  CC���z�������{CFǮ                                    Bx����  p          @�G���(���\���IG�CD!H��(��z�n{��CF�                                    Bx���n  �          @Ϯ��=q���
��
=�K�CDY���=q���p�����CF�                                    Bx���  �          @����G���\��  �-G�CC�3��G��녿E�����CE�
                                    Bx���  �          @Ӆ�ʏ\��
=�h����(�CB�
�ʏ\���;���  CDJ=                                    Bx�� `  �          @��H�ə��޸R�Y����CCu��ə���33�Ǯ�[�CD�                                    Bx��  �          @�(��ʏ\��G��p���CC�\�ʏ\��Q������\CE�                                    Bx���  �          @Ӆ��=q��(��p���\)CC33��=q��33���H����CD�3                                    Bx��,R  �          @��H���ÿ�p��}p��
�RCCaH���ÿ��
=q��{CD�q                                    Bx��:�  �          @�33��G���p���G��Q�CCaH��G����������CE�                                    Bx��I�  �          @ָR���
��\�����\)CC�����
���R�+���{CE^�                                    Bx��XD  �          @�  ���Ϳ���z���\CCǮ������\�0�����HCE�f                                    Bx��f�  �          @ָR�ə���׿�=q�6ffCD�)�ə���ÿTz����
CF��                                    Bx��u�  �          @�\)��G����H��z��Ap�CEL���G��\)�fff��z�CG��                                    Bx���6  �          @�\)�ə�������{�:�\CE.�ə��{�Y�����CGff                                    Bx����  �          @ָR��Q���H����?33CE^���Q��\)�aG����CG��                                    Bx����  T          @׮�ə���
=����>�HCE\�ə��p��c�
��\CGT{                                    Bx���(  �          @����ff���ÿ�=q�\)CDQ���ff�G��(����
CF�                                    Bx����  �          @�33���
��ff�(����  CB33���
��zᾏ\)�\)CC0�                                    Bx���t  �          @ʏ\��z΅{�333�˅C@����z῾�R��Q��O\)CA�H                                    Bx���  �          @�=q���H��  �B�\��CA�{���H��녾Ǯ�`��CC�                                    Bx����  �          @����G���
=�:�H���
CC����G���ff���R�5�CD��                                    Bx���f  �          @ʏ\�����У׿Q���CC��������
����n�RCDaH                                    Bx��  �          @�=q���׿У׿�G���CC����׿��ÿ(���  CD�\                                    Bx���  �          @�������{�E���33CB�������  ��p��R�\CCٚ                                    Bx��%X  �          @˅��z�Ǯ�z���
=CBB���z��33�L�Ϳ�ffCC\                                    Bx��3�  p          @�G���녿���8Q��љ�CBG���녿���{�G
=CC^�                                    Bx��B�  �          @ȣ����ÿ��H�c�
��HCA�f���ÿУ׿���33CC!H                                    Bx��QJ  �          @����������׿s33�
�RC@�������Ǯ�����Q�CB�                                     Bx��_�  
�          @�\)��Q쿮{�Y����(�C@�q��Q�\������CB+�                                    Bx��n�  �          @�{��\)��(���  ���C?}q��\)��z�333��ffCAG�                                    Bx��}<  �          @����R����z�H�p�C@ff���R��  �&ff���CB)                                    Bx����  �          @���ff��\)�xQ���
C@���ff��ff�!G����CB�
                                    Bx����  �          @���{���R�Tz���ffCB���{�У׾�����
CCc�                                    Bx���.  �          @�{���Ϳ��H�@  ��  CD!H���Ϳ�=q��33�O\)CE@                                     Bx����  �          @�ff����33�5���HCC������\���
�?\)CD��                                    Bx���z  �          @�
=���R��
=�#�
��CC�=���R�����  �33CD��                                    Bx���   �          @�(���녿�녿.{��CD�����   ��  ��CE��                                    Bx����  �          @У��������  �G�CD������ �׿\)��\)CF
                                    Bx���l  p          @�����ÿ�녿��R����CC33������\�����^{CF��                                    Bx��  
�          @�  �ƸR���ÿ�z���\)CDQ��ƸR�(����H�G�CGu�                                    Bx���  �          @׮��Q����
=�f�RCDW
��Q��
=q��p��(��CG�                                    Bx��^  �          @ָR��Q���
�У��a�CC޸��Q�������%�CF}q                                    Bx��-  �          @�
=���H�޸R����8��CCQ����H���R�n{��\)CEk�                                    Bx��;�  �          @�p���=q��zῧ��5��CB����=q��z�k���p�CD�=                                    Bx��JP  �          @�����녿�zῧ��5G�CB��녿�33�k����CD�\                                    Bx��X�  �          @�z���녿У׿���3�CBs3��녿�\)�h����z�CD}q                                    Bx��g�  �          @�p��ʏ\�˅��{�;�CB#��ʏ\���Ϳz�H�33CDG�                                    Bx��vB  �          @�{��녿�(���\)�>{CC8R��녿�(��xQ��=qCEY�                                    Bx����  �          @��ə���\)���R�NffCBp��ə���33��{��CD�\                                    Bx����  �          @������У׿�=q�^ffCB�
�����
=��Q��'�CEff                                    Bx���4  �          @������
�˅����h(�CB�����
��33��G��2=qCE@                                     Bx����  �          @�Q����
����(��PQ�CC@ ���
��Q쿊=q���CE��                                    Bx����  �          @�G���(���\�����Mp�CD���(��녿�ff��CFW
                                    Bx���&  �          @�G���p��ٙ�����<��CC^���p���Q�s33��CEp�                                    Bx����  �          @�����Ϳ�\��Q��JffCD������녿���G�CF=q                                    Bx���r  �          @�=q��
=��
=��ff�7
=CC!H��
=���k��G�CE!H                                    Bx���  �          @ҏ\��  ��녿�(��*�RCB����  ��{�Y����p�CD��                                    Bx���  �          @����ʏ\��zΐ33�\)CB�3�ʏ\��{�G���  CDc�                                    Bx��d  �          @������ÿ�ff���H�(Q�CD����� �׿Q��ᙚCEǮ                                    Bx��&
  �          @�z��Ǯ���ÿ�{�<z�CDJ=�Ǯ��
�s33�  CFJ=                                    Bx��4�  �          @��H��p����H���
�W�CC����p����R��33�!�CE޸                                    Bx��CV  �          @��H���Ϳ�
=���j�RCCL����Ϳ��R����4��CE��                                    Bx��Q�  T          @��H��z���H��Q��mCC�)��z��G�����7
=CF=q                                    Bx��`�  �          @����{��Q��G��t��CCB���{� �׿����?
=CE��                                    Bx��oH  �          @�����ff��p�����d(�CC����ff�G���  �.{CF\                                    Bx��}�  �          @��
���Ϳ�=q�޸R�t��CBaH���Ϳ�녿����B=qCE)                                    Bx����  �          @ҏ\�\���R������C?�\�\��33������CC8R                                    Bx���:  �          @�33��(���33�G���=qC@�
��(���\�ٙ��nffCD)                                    Bx����  �          @љ����
���H���
�|z�CAaH���
���
�����Mp�CD5�                                    Bx����  �          @љ����Ϳ�(�����h��CAxR���Ϳ�\�����:{CD�                                    Bx���,  �          @�33��  ��p����
�V�RCAL���  ��  ���H�)�CC�f                                    Bx����  �          @��
�ȣ׿��R��G��S33CAff�ȣ׿�G���Q��%p�CC��                                    Bx���x  �          @�(���G���p���p��N�RCA@ ��G��޸R��z��!CCz�                                    Bx���  �          @����H���
���C�
CA�{���H���
����ffCC��                                    Bx���  �          @�33�ə����R��ff�5CAY��ə���(��z�H�	G�CCE                                    Bx��j  �          @ҏ\�ƸR����33�Dz�CC��ƸR��zῆff�
=CE{                                    Bx��  �          @�p����
��z῎{���CB�)���
���E����HCD!H                                    Bx��-�  �          @�\)��z���ÿ����CC���z���R�0������CEQ�                                    Bx��<\  T          @ָR���
�����
��CD����
�   �(����z�CEk�                                    Bx��K  �          @�{��33��녿s33�{CD�=��33�녿���z�CE�q                                    Bx��Y�  �          @�p����H��33�aG���G�CD�����H�녿   ��G�CE�q                                    Bx��hN  �          @�z���=q��33�@  ��  CD��=q� �׾\�Q�CE�f                                    Bx��v�  �          @�ff���Ϳ��Ϳ0�����
CD
���Ϳ�Q쾨���2�\CD�H                                    Bx����  �          @�
=��ff��G��&ff��=qCCG���ff���;��R�)��CD�                                    Bx���@  �          @�  ��ff��녿#�
��CDJ=��ff��(���\)�CD�q                                    Bx����  �          @�\)�����Tz����
CC������
=�����
CD�q                                    Bx����  �          @�\)��ff���}p��
=CB����ff��=q�(����z�CC�\                                    Bx���2  �          @�ff���
��ff�z�H��RCCǮ���
�����!G����CE                                    Bx����  �          @�\)��33��J=q�ָRCF.��33��;Ǯ�U�CG�                                    Bx���~  �          @ָR��(����Ϳs33��CD#���(����R�
=���\CEQ�                                    Bx���$  �          @�{��33�������{CC��33���H�=p���z�CE(�                                    Bx����            @����ƸR���Ϳ�\�w�CBz��ƸR��33�����I�CE                                      Bx��	p  �          @������޸R���H�n{CC���G���{�=G�CF!H                                    Bx��  �          @׮��(��Q��=q�{�CG0���(�����z��A��CI��                                    Bx��&�  �          @����ƸR����=q�X(�CG\)�ƸR����z��ffCI^�                                    Bx��5b  �          @أ���\)�
=q�Ǯ�Tz�CG(���\)�=q�����CI!H                                    Bx��D  �          @�G����H�   ����<z�CExR���H�p���G����CG8R                                    Bx��R�  �          @�
=����׿����5��CH����p��fff��ffCI�R                                    Bx��aT  �          @�
=���H�\)����8  CJ=q���H�,(��aG����CK�\                                    Bx��o�  �          @ָR�����(�ÿ���1�CK�f�����4z�L����z�CM)                                    Bx��~�  �          @�p���z��\)��ff�3�CH\��z����aG����
CI�H                                    Bx���F  T          @�G���33�33��G��+�CE����33�\)�c�
��Q�CGk�                                    Bx����  �          @�����33��
���H�$z�CE�3��33��R�W
=��\CGff                                    Bx����  >          @أ���G���Ϳ��H�$(�CGG���G����O\)��33CH��                                    Bx���8            @�\)��33��p��xQ����CEL���33����R��Q�CFn                                    Bx����            @ָR��z��\)�W
=���CDG���z��p��   ���CE:�                                    Bx��ք  �          @�{���
���������CE����
�G��aG���
=CE�)                                    Bx���*  �          @ָR���Ϳ��H����{CE������G��8Q���
CE��                                    Bx����  �          @�
=��(���
��p��H��CE�H��(���#�
��CF#�                                    Bx��v  T          @�{��(��������H���RCE���(��   �������CEu�                                    Bx��  �          @���θR�\�
=q��p�CA=q�θR�˅��\)���CA�{                                    Bx���  T          @�(���p��Ǯ���H����CA�H��p���\)�k����HCB#�                                    Bx��.h  �          @ָR�θR��  ����~�RCC+��θR��ff�.{��
=CC��                                    Bx��=  T          @أ���
=��p�������CE
=��
=��p�>L��?�z�CE                                      Bx��K�  
�          @�  ��G���\)�����0��CA����G���33�L�;�CB&f                                    Bx��ZZ  T          @�\)���ÿ�=q����]p�CA�����ÿ�\)�\)���HCA�3                                    Bx��i   �          @�ff��  ��  ������C@����  ���þ����$z�CA�\                                    Bx��w�  �          @���У׿�{=�Q�?=p�C?���У׿�=q>��R@(��C?��                                    Bx���L  �          @�(��Ϯ����>�?���C?��Ϯ���>�{@>�RC?p�                                    Bx����  �          @�(���Q쿠  <��
>�C>�f��Q쿞�R>aG�?��C>�q                                    Bx����  �          @�z���G���=q�B�\��
=C=ff��G����ͼ��
���C=��                                    Bx���>  �          @��Ӆ�E��z�����C:���Ӆ�Y����ff�y��C;\)                                    Bx����  �          @�(��љ��#�
�@  ��G�C9���љ��@  �#�
��=qC:�=                                    Bx��ϊ  �          @ҏ\������ÿ(������C?�)�����33��G��w�C@^�                                    Bx���0  �          @�G��˅��������G�C@�H�˅�\��\)�p�CAk�                                    Bx����  
�          @�Q��ƸR�޸R�c�
���CC�f�ƸR��{������\CD��                                    Bx���|  �          @�{���
�Ǯ�ff���\CB�f���
��\)��=q��=qCE�                                    Bx��
"  
�          @�ff�����33���
�:�HCC������>8Q�?���CB�3                                    Bx���  �          @�ff��ff������
��C?����ff��
=�Q��陚CA                                      Bx��'n  �          @�\)��=q�n{��(����\C<����=q��(�����{C?ff                                    Bx��6  �          @��H��
=��{��\��\)C=����
=��
=��{�;�C>L�                                    Bx��D�  �          @�  ��zῊ=q�����p�C=�{��zῑ녾����*=qC>�                                    Bx��S`  �          @��H��{���H��p���CG���{�\)��z��r�RCI��                                    Bx��b  �          @�=q��\)��33�������CF\)��\)�
=q���
�`��CH��                                    Bx��p�  �          @�{��G�����Q����HCE5���G��ff��=q����CG��                                    Bx��R  �          @�=q�������
=��Q�CE
=���
=����z�CG��                                    Bx����  �          @�33��33��  ��G��0��C?#���33��
=����Q�C@�f                                    Bx����  �          @�p��У׿k���  �
{C<
=�У׿���Y�����HC=:�                                    Bx���D  �          @�ff��\)��\)�Y����=qC?����\)���R�#�
��\)C@�                                    Bx����  �          @�33��{��=q��\)�c�CBJ=��{��ff��\)�?�CD33                                    Bx��Ȑ  T          @׮�ȣ׿�=q��{���\CB!H�ȣ׿���{�]p�CDQ�                                    Bx���6  �          @�\)�ȣ׿��
������CA�q�ȣ׿�ff��
=�fffCD�                                    Bx����  �          @�ff��33�˅�����9�CB���33��\�����\)CC�\                                    Bx���  �          @�{��{��G��
�H���C?u���{�Ǯ��(���
=CB!H                                    Bx��(  �          @�
=��(������
=�Ep�CA�{��(���p������$(�CC0�                                    Bx���  �          @�z���{������G��s�
C@����{��  �k����HCA)                                    Bx�� t  �          @�G����
��ff>���@a�C?�\���
��p�?z�@�z�C>�                                    Bx��/  �          @�ff��{����?�=qA>�HC>�R��{��  ?��RAU�C=&f                                    Bx��=�  �          @�  ��  ��{?���AD��C>
��  �fff?��
AYC<:�                                    Bx��Lf  �          @θR�ƸR�s33?��A[�
C<���ƸR�8Q�?�33AmG�C:�f                                    Bx��[  �          @θR��
=�h��?��RAT��C<L���
=�0��?���Aep�C:O\                                    Bx��i�  �          @�
=�ƸR�J=q?��Aj=qC;5��ƸR���?�p�Ax(�C9�                                    Bx��xX  �          @�
=�Ǯ�xQ�?���AEC<�)�Ǯ�E�?�G�AW�C;�                                    Bx����  
�          @Ϯ�ə����?��A{C=�{�ə��n{?�p�A.ffC<c�                                    Bx����  �          @Ϯ��  �u?���AN�RC<�q��  �@  ?���A`(�C:�
                                    Bx���J  �          @�����Q�^�R?�\)Af=qC;�H��Q�!G�?�p�Aup�C9Ǯ                                    Bx����  �          @љ���G�����?���AL  C=���G��c�
?�=qA_�
C<{                                    Bx����  �          @�=q��Q쿨��?�p�APQ�C?�f��Q쿌��?��Ah(�C>                                      Bx���<  �          @ҏ\��
=����?�=qA^=qCA(���
=��p�?�G�Ax��C?&f                                    Bx����  �          @���Ǯ����?��AY��C?��Ǯ����?��HAqG�C=�R                                    Bx���  �          @�����ff���?�p�AQ��C@����ff��?�33Aj�RC>�R                                    Bx���.  �          @�����\)��\)?��AD��C@^���\)��?ǮA]��C>�)                                    Bx��
�  �          @�Q���
=����?��A=�C@xR��
=��
=?�G�AV�RC>Ǯ                                    Bx��z  �          @�����
=��33?��AE�C@�3��
=����?���A^=qC>��                                    Bx��(   �          @Ϯ��(��ٙ�?��A8  CC���(��\?�G�AW
=CA��                                    Bx��6�  �          @У����Ϳ�?�G�AV�HCA����Ϳ��H?�
=ApQ�C?�                                    Bx��El  �          @�Q��ƸR��?��A7
=C@�f�ƸR���R?��HAPQ�C?L�                                    Bx��T  �          @�  ���Ϳ�(�?�  AU��CAn���Ϳ�G�?�
=Ao�C?�{                                    Bx��b�  �          @Ϯ����z�?�33AG33C@ٚ�����H?���A`(�C?!H                                    Bx��q^  �          @Ϯ��ff���?�{AA��C?����ff����?\AX��C>J=                                    Bx���  �          @�
=��ff��{?���A*�\C@ff��ff����?�{AB�\C>�                                    Bx����  �          @�
=��{��Q�?�z�A%��CA{��{���
?��A>�HC?��                                    Bx���P  �          @�\)��Q쿘Q�?�A&ffC>�=��Q쿃�
?��A:�HC=^�                                    Bx����  �          @�\)��  ���?��\AC?ٚ��  ��?�
=A(z�C>�)                                    Bx����  �          @θR��  ����?�Q�A)C=�q��  �h��?��A<(�C<O\                                    Bx���B  �          @θR��  ��33?���A�C>k���  ��  ?��RA0��C=�                                    Bx����  �          @�{��G�����?Y��@��
C=�f��G��s33?z�HAQ�C<��                                    Bx���  �          @�{��=q��ff?(�@��C=u���=q�xQ�?:�H@љ�C<��                                    Bx���4  �          @���Q쿣�
?:�H@У�C?�
��Q쿗
=?aG�@���C>�3                                    Bx���  �          @�{��  ��=q?L��@�C@���  ��(�?uA��C?�                                    Bx���  �          @�
=��\)���
?L��@�=qCA�{��\)��?z�HA\)C@޸                                    Bx��!&  �          @θR��\)���
?.{@��CAǮ��\)��
=?\(�@�{C@��                                    Bx��/�  �          @�ff��  ��(�?z�@�p�CA=q��  ���?B�\@׮C@�                                    Bx��>r  �          @�{�ȣ׿�p�?J=q@��C?\�ȣ׿�\)?n{A��C>!H                                    Bx��M  �          @θR�ȣ׿�p�?xQ�A	p�C?��ȣ׿�{?�{A�C>�                                    Bx��[�  �          @��H���Ϳ��R?xQ�A  C?n���Ϳ�\)?�{A ��C>L�                                    Bx��jd  �          @˅���
��p�?p��A(�CA�����
����?�{A ��C@xR                                    Bx��y
  "          @��H��������?��
A=qCB�\������Q�?��HA0��CA\)                                    Bx����  �          @��
��=q�У�?�  A��CC\��=q��  ?�Q�A,  CA��                                    Bx���V  �          @��H��=q��(�?�ffAG�CA�H��=q���?�(�A1C@k�                                    Bx����  �          @��H�\���?�33A&�\C@�f�\��  ?�ffA=��C?��                                    Bx����  �          @���{���?�
=A)p�C?���{��z�?�=qA>ffC>��                                    Bx���H  �          @�p���33����?�z�AJffCAY���33���\?���AaC?�=                                    Bx����  �          @�33���R��
=?�
=At��CA}q���R��p�?�=qA�{C?��                                    Bx��ߔ  "          @�(���G���z�?��HAT��CA���G���p�?�\)Ak\)C?z�                                    Bx���:  �          @����녿�z�@�
A��C?W
��녿k�@�A�
=C=                                      Bx����  �          @˅���H�B�\@<(�A�C;����H��
=@@  A�G�C8O\                                    Bx���  �          @�33���
��ff@.�RA�z�C>�H���
�=p�@5AԸRC;u�                                    Bx��,  �          @ʏ\��  ��33@p�A��CA�f��  ���@
=A��C?.                                    Bx��(�  �          @��
���
���?�Q�Av�RCD�R���
�˅?��A�CC#�                                    Bx��7x  �          @�
=��p���33?�\A~�RCEǮ��p���Q�?�p�A�ffCC�f                                    Bx��F  T          @���p��޸R?��A�p�CDW
��p����
?�(�A��CBp�                                    Bx��T�  �          @�{��ff��G�?ٙ�Au��CD����ff�Ǯ?��A���CB��                                    Bx��cj  �          @�{����=q?�Q�A�  CB޸������@ffA�=qC@�{                                    Bx��r  �          @����\)���
@"�\A��HC@�)��\)�}p�@*=qA���C=�{                                    Bx����  �          @�{������R@1�A���C@T{����k�@9��A�z�C=E                                    Bx���\  �          @θR�����  @+�A�  C@B�����s33@333AͅC=\)                                    Bx���  �          @θR������@(��A���CD33�������@333AͮCAaH                                    Bx����  T          @�
=��
=��@	��A��
CG���
=��=q@�A�  CE��                                    Bx���N  T          @�z�����{@�HA�Q�CC�{�����@%�A���CAE                                    Bx����  �          @ə����׿�  @(��A�ffCC@ ���׿��H@2�\A�=qC@aH                                    Bx��ؚ  �          @ə���G�����@'�A���CB����G���z�@0��A�(�C?ٚ                                    Bx���@  �          @����(��\@�A��RCC&f��(���G�@$z�A�z�C@�{                                    Bx����  �          @�����Ϳ˅@��A��RCC�q���Ϳ��@�HA�
=CA^�                                    Bx���  �          @�(���G�����@.�RA��HCC�\��G����\@8Q�A���C@�                                    Bx��2  �          @���������?�G�A�  CB��������?�A��C@�f                                    Bx��!�  �          @�p���33�޸R?n{ACC���33���?�{A�CC                                      Bx��0~  �          @�p���ff��(�?�
=ArffCD���ff���
?���A��
CBp�                                    Bx��?$  �          @�����  �޸R?�{AC�
CD!H��  �˅?��
A]G�CB��                                    Bx��M�  �          @�����
����?�A��CE8R���
��\)?��RA���CCp�                                    Bx��\p  �          @��
��33��
=?��A��CD���33��p�@33A�{CB0�                                    Bx��k  �          @��
��  �˅@�RA�=qCCu���  ����@Q�A��CA=q                                    Bx��y�  �          @�(���{��\)@(�A�ffCF&f��{�У�@�A�=qCD�                                    Bx���b  �          @�������\@ffA�CG������@33A��RCE�                                    Bx���  �          @�p���(���33?�Q�A�G�CC�R��(�����@ffA��CAٚ                                    Bx����  �          @�{��  �G�?�G�A~{CI�\��  ��?�p�A�p�CG�                                    Bx���T  �          @���  ��?�A���CHB���  ��@�A�Q�CFs3                                    Bx����  �          @�ff���H��@ffA�(�CKz����H���@�A�CI�=                                    Bx��Ѡ  �          @љ���z��8��@z�A�=qCP.��z��(��@&ffA��CN�                                    Bx���F  �          @�z����R�&ff@
=qA��
CL� ���R�Q�@=qA�{CJ�
                                    Bx����  �          @��H����&ff@33A��HCL������
=@"�\A�G�CJٚ                                    Bx����  �          @�(����\�=p�@'�A��CQ����\�,(�@8��A�
=CN�                                    Bx��8  �          @����H�>�R@+�A���CQ.���H�-p�@<��A�{CN޸                                    Bx���  �          @�{��
=�AG�@5A�  CR\��
=�.�R@G�Aߙ�CO��                                    Bx��)�  �          @�=q��{���@1�A�
=CD���{��{@:�HA�ffCAp�                                    Bx��8*  �          @�G�������H@EA��CET{�����z�@O\)A�{CB^�                                    Bx��F�  �          @�  ��p����@B�\A��HCG@ ��p��˅@L��A�ffCD\)                                    Bx��Uv  �          @ҏ\���R��@3�
A���CK(����R�33@AG�A�\)CH��                                    Bx��d  �          @��
�����@H��A�
=CL������
@W
=A��CI0�                                    Bx��r�  	�          @����G���H@333AƸRCK�\��G����@@��A�G�CI)                                    Bx���h  T          @����=q�\)@8��A�\)CI���=q���H@EAܸRCGc�                                    Bx���  
Z          @���=q��@>{A�
=CIc���=q���@J=qA��CF�                                    Bx����  
Z          @��
������@<��A��
CF�)�������@G
=A��CC��                                    Bx���Z  T          @�  ��G����\@L��A�RC>ff��G��5@Q�A�\)C;T{                                    Bx���   "          @љ���{���H@0  AƸRCD����{��Q�@9��A�{CB=q                                    Bx��ʦ  �          @��H��33����@%A�p�CC��33����@.�RAîC@�q                                    Bx���L  T          @�33���\��33@(�A��CF����\��z�@&ffA��CC�                                    Bx����  
�          @����{��
=@(Q�A��CF���{��
=@333A�Q�CDxR                                    Bx����  
�          @ҏ\��ff���@,��A�{CF=q��ff��\)@7
=A�z�CC��                                    Bx��>  "          @�=q��{��\)@,(�A�p�CF0���{��\)@6ffAͮCC�)                                    Bx���  
�          @������33@*�HA�  CF� ����33@5�A�z�CD33                                    Bx��"�  �          @����p���Q�@*�HA�Q�CFٚ��p���Q�@5A�
=CD�\                                    Bx��10  T          @����\)��
=@#�
A��
CF�
��\)��Q�@.�RA�Q�CDk�                                    Bx��?�  "          @����ff��@'
=A���CF����ff��@1G�A�  CD\)                                    Bx��N|  
          @ҏ\��p��   @*=qA��CGh���p���  @5�A�{CE(�                                    Bx��]"  |          @��H���R���R@(Q�A�=qCG.���R�޸R@333A��HCD��                                    Bx��k�  
�          @��H���z�@(Q�A�=qCH
=����=q@333A�p�CEٚ                                    Bx��zn  
Z          @�=q��33���@,(�A�CH����33���@7�AυCF��                                    Bx���  "          @�������	��@1�A�(�CI#��������@<��A�  CF�{                                    Bx����  "          @љ�����
�H@5�A��HCI�������z�@@��A��HCG+�                                    Bx���`  "          @�=q���\�33@E�A߮CKT{���\�G�@QG�A�RCH                                    Bx���  T          @�=q�����@G�A��CK:�����   @S�
A�{CH��                                    Bx��ì  �          @�G����(�@U�A�\)CJ������33@`��B��CH)                                    Bx���R  
          @љ��������@b�\B�HCLE���׿���@n{B
�CI5�                                    Bx����  
�          @����Q��ff@aG�B��CM0���Q���\@mp�B	�CJ33                                    Bx���  �          @�G���\)�(��@i��B��CQ+���\)�z�@w�B�CN�                                    Bx���D  �          @љ���(��%@a�BG�CO�R��(���@n�RB
=CM
=                                    Bx���  
�          @����G��%@i��B=qCPh���G��G�@w
=B�CM\)                                    Bx���  �          @�Q���G��%@dz�B�HCPk���G���@qG�B�RCMs3                                    Bx��*6  "          @�
=��ff�+�@c33B(�CQ����ff��@p��B\)CNǮ                                    Bx��8�  T          @�
=�����,��@[�B {CQ��������@h��B	33CN��                                    Bx��G�  �          @Ϯ���#33@VffA��HCOY������@c33B�CL�H                                    Bx��V(  �          @θR��ff��R@Tz�A�CN����ff�(�@aG�B{CK�
                                    Bx��d�  �          @�{��\)�   @O\)A�CN�H��\)�{@\(�B  CL                                      Bx��st  �          @���\)�z�@UA�Q�CL����\)��@aG�B��CJ8R                                    Bx���  
Z          @�����\��@0  A�=qCJE���\����@:�HA�  CH�                                    Bx����  
�          @θR��Q��ff@8Q�A���CL��Q��ff@Dz�A�CI�                                     Bx���f  �          @θR���R�ff@-p�A���CI����R��\)@8Q�A��
CF��                                    Bx���  �          @�\)��
=�
�H@-p�AŮCI�)��
=��Q�@8Q�A���CG�                                     Bx����  T          @Ϯ��ff�
=q@333A�Q�CI����ff��
=@>{AمCGz�                                    Bx���X  "          @љ���(��33@*=qA���CH  ��(�����@4z�A��
CE��                                    Bx����  �          @�=q��Q��(�@>�RA�  CG����Q��(�@HQ�A��
CEc�                                    Bx���  
�          @��H��33��@1�A�CHc���33����@<(�A�{CFG�                                    Bx���J  "          @�z���  ���R@&ffA��CG
��  ��\@0  AĸRCE#�                                    Bx���  �          @Ӆ��(��   @3�
A�33CG�\��(���\@=p�A���CEn                                    Bx���  �          @��
��z��z�@G�A��HCF  ��z��p�@�HA�CDT{                                    Bx��#<  �          @��
���Ϳ��
?�z�Ahz�CD+����Ϳ��?�ffA|Q�CB�R                                    Bx��1�  �          @���Ǯ���
?��\A2�RCA� �Ǯ��?��AC�C@�{                                    Bx��@�  
�          @У���=q���
?n{A�
C?z���=q���H?��
A�C>��                                    Bx��O.  �          @љ���=q��z�?��\A33C@����=q����?���A�\C?�\                                    Bx��]�  "          @У��ȣ׿��H?�  A��CA
�ȣ׿�\)?�{A��C@\)                                    Bx��lz  �          @Ϯ��  ���R?p��A�CAs3��  ��z�?��A=qC@                                    Bx��{   
Z          @�\)��\)��p�?xQ�A	p�CAW
��\)��33?�=qAC@�H                                    Bx����  �          @�p���\)��  ?^�R@���C?\)��\)��
=?xQ�A
�\C>��                                    Bx���l  "          @�p��Ǯ���\?J=q@�C?z��Ǯ����?c�
@�\)C>��                                    Bx���  �          @�p���ff����?uA	�C@���ff���R?��A�C?Q�                                    Bx����  �          @�  ��녿��H?n{AQ�C>�)��녿���?��
AG�C>0�                                    Bx���^  �          @�\)���ÿ�  ?z�HA
=C?G����ÿ�?���Az�C>�{                                    Bx���  �          @�Q��ʏ\��
=?h��A Q�C>���ʏ\��{?�  A��C=�                                    Bx���  �          @����33���\?���A�\C?Y���33��Q�?�A$  C>�
                                    Bx���P  �          @�33������\?h��@��C?.�����Q�?�  A  C>��                                    Bx����  "          @љ���zῚ�H?J=q@�z�C>�R��zΐ33?aG�@�{C>+�                                    Bx���  
�          @Ӆ��
=��Q�?8Q�@���C>p���
=����?O\)@ᙚC=�                                    Bx��B  �          @��
��\)��?(��@�ffC>8R��\)��\)?=p�@θRC=                                    Bx��*�  "          @�(���  ����?�R@�33C=�
��  ��=q?333@\C=h�                                    Bx��9�  
�          @�(��θR���R?G�@׮C>�f�θR��
=?^�R@�C>^�                                    Bx��H4  
�          @��H����  ?&ff@�(�C>�q������?=p�@�{C>�=                                    Bx��V�  �          @љ���ff��G�>�@�ffC<�)��ff�xQ�?��@��C<�                                    Bx��e�  �          @���Ϯ�L��>�@�\)C;
=�Ϯ�B�\?
=q@�  C:�3                                    Bx��t&  T          @�Q���{�Y��>�(�@r�\C;�=��{�Q�>��H@�33C;=q                                    Bx����  T          @�����ff�n{>�(�@s33C<:���ff�fff?   @���C;�                                    Bx���r  �          @ҏ\��\)���
>�(�@p  C=���\)��  ?�\@��C<�                                    Bx���  T          @љ������H>�\)@p�C>������
=>�p�@O\)C>h�                                    Bx����  "          @��H�����
?(��@��C?J=����p�?@  @��C>�
                                    Bx���d  �          @��H���Ϳ�33?�R@�p�C@L����Ϳ���?8Q�@��C?޸                                    Bx���
  "          @�33�Ϯ�k�?=p�@���C<
�Ϯ�\(�?L��@�\)C;�
                                    Bx��ڰ  
�          @��
��G��&ff?:�H@��
C9�3��G����?G�@أ�C95�                                    Bx���V  �          @�(���  ����?
=@��C=�f��  ���?+�@��\C=�                                     Bx����  "          @�33��\)��z�?\)@�(�C>!H��\)��{?#�
@��C=�                                     Bx���  "          @��H��\)���>�@�
=C=�=��\)��ff?\)@��C=33                                    Bx��H  �          @�(��Ϯ���>�{@<(�C?B��Ϯ��G�>�(�@p��C?                                    Bx��#�  �          @Ӆ��=q��R>�33@E�C9\)��=q�
=>���@]p�C9�                                    Bx��2�  4          @ҏ\��Q�h��>aG�?�
=C;�q��Q�c�
>�z�@ ��C;�{                                    Bx��A:  �          @ҏ\�Ϯ���
>aG�?��C<�R�Ϯ��G�>�z�@!G�C<�\                                    Bx��O�  �          @Ӆ��녿E��#�
�uC:�3��녿E�=L��>�(�C:�                                    Bx��^�  �          @��H��Q�}p�����  C<����Q�}p�=#�
>\C<��                                    Bx��m,  �          @�=q��  �p��    �#�
C<B���  �p��=�\)?z�C<=q                                    Bx��{�  �          @����Q�u�#�
���C6&f��Q쾔z��R��{C6�\                                    Bx���x  �          @�
=����Ǯ�333��ffC7�������G��+���C7�3                                    Bx���  �          @У���{��ff�L�����C8  ��{��\�E���  C8�                                     Bx����  �          @љ����@  ��R��  C:�����L�Ϳ���Q�C;�                                    Bx���j  �          @�����p����
���R�0  C=)��p���ff�u�C=J=                                    Bx���  �          @�����ff��R�J=q���C9z���ff�+��=p���  C9�R                                    Bx��Ӷ  �          @Ϯ�ə���Q�=�?��C>���ə���>L��?�ffC>��                                    Bx���\  �          @�
=��ff��{?O\)@�p�CB�\��ff��ff?k�A
=CB                                    Bx���  �          @�
=�ȣ׿\?�@��RCA���ȣ׿�p�?#�
@�p�CAG�                                    Bx����  �          @�Q��˅��=q>�{@?\)C?ٚ�˅���>�(�@u�C?�)                                    Bx��N  �          @�\)��\)��{?#�
@�z�CBz���\)�Ǯ?@  @��CB�                                    Bx���  �          @Ϯ���ÿ�ff�aG���Q�CA�)���ÿǮ��G��s33CA��                                    Bx��+�  �          @Ϯ��p����H?k�A
=CC����p����?��AQ�CB�                                    Bx��:@  �          @љ���p�����?�z�A#33CD�R��p���\?��A5��CC�R                                    Bx��H�  �          @Ϯ�Å��z�?�ffACEO\�Å��=q?�Q�A(��CD�H                                    Bx��W�  �          @�  �\�z�?}p�A�CF�\���R?���A z�CF�                                    Bx��f2  �          @�Q�������R?uAz�CHE�����	��?�\)A�RCG��                                    Bx��t�  �          @�Q���(���p�?fff@�\)CE�)��(���z�?��A\)CEE                                    Bx���~  �          @�Q��Å����?��ACE���Å��\)?�Q�A)�CD�q                                    Bx���$  �          @�  ���
��Q�?��\A�
CE�H���
��\)?�33A#33CD�R                                    Bx����  �          @�  ��z��33?p��A��CE5���z��=q?�=qAz�CD�
                                    Bx���p  �          @�\)���H��
=?���AQ�CE�)���H����?��HA+�CD�                                    Bx���  �          @�{���Ϳ��?
=q@��CDB����Ϳ�  ?(��@�CC��                                    Bx��̼  �          @�p�������\?G�@�
=CF�����ÿ�p�?k�A(�CF.                                    Bx���b  �          @�������   >�G�@|(�CFG���녿�(�?z�@�ffCE��                                    Bx���  �          @�z��\����?�R@��CD�3�\��ff?@  @���CD��                                    Bx����  �          @�
=�����\)?
=@��RCD�H�������?8Q�@�(�CD}q                                    Bx��T  �          @Ϯ������H?�@���CE�������z�?5@ȣ�CEE                                    Bx���  �          @θR�ƸR��(�?   @�ffCC}q�ƸR��
=?�R@���CC&f                                    Bx��$�  �          @�{��ff��Q�>�@��\CCB���ff��33?z�@�z�CB�3                                    Bx��3F  �          @���
=����>B�\?ٙ�CBc���
=��=q>���@-p�CB=q                                    Bx��A�  �          @�
=�ə�����>��R@0  C@���ə���
=>��@i��C@                                    Bx��P�  �          @Ϯ�˅����>\@W�C>���˅��>�@��C>h�                                    Bx��_8  �          @�  ���
��ff�#�
��\)C?����
���=��
?=p�C?}q                                    Bx��m�  �          @�
=�ə���
=�����'�C@� �ə���Q�B�\��(�C@�f                                    Bx��|�  �          @θR���H���׾�33�G�C>!H���H��zᾊ=q�=qC>Q�                                    Bx���*  �          @��ʏ\���
��Q��L(�C=:��ʏ\��ff��\)�!�C=n                                    Bx����  �          @�
=��(����
������C=#���(����
��\)�(��C=33                                    Bx���v  �          @θR��(����\<#�
=���C=\��(���G�=��
?:�HC=�                                    Bx���  T          @˅�ȣ׿���B�\��p�C=�f�ȣ׿�=q����ffC=�q                                    Bx����  �          @�����=q�p��>aG�?�p�C<p���=q�k�>�z�@$z�C<G�                                    Bx���h  �          @�����=q�xQ����z�C<����=q�xQ�=#�
>��
C<��                                    Bx���  �          @�{���
�u>k�@�\C<�����
�p��>���@(Q�C<^�                                    Bx���  �          @�
=����c�
=��
?333C;�����aG�>\)?�  C;ٚ                                    Bx�� Z  �          @�
=����W
=>��@33C;� ����Q�>��
@4z�C;Q�                                    Bx��   �          @Ϯ��{�+�>�
=@p  C9�3��{�#�
>��@��C9��                                    Bx���  �          @Ϯ���L��>��
@2�\C;����G�>�p�@Q�C:�H                                    Bx��,L  �          @�ff��z�Q�>���@;�C;Q���z�L��>Ǯ@\(�C;�                                    Bx��:�  �          @θR��z�O\)>��@�C;:���z�E�?�@�C:��                                    Bx��I�  �          @�p��˅�333>��H@��
C:Q��˅�+�?
=q@��C:                                      Bx��X>  �          @θR���H�p��?L��@�\C<n���H�aG�?\(�@��C;�f                                    Bx��f�  �          @θR��=q�}p�?c�
@�z�C<����=q�n{?uA  C<T{                                    Bx��u�            @�ff�ʏ\�Q�?s33A(�C;\)�ʏ\�@  ?�G�A(�C:�q                                    Bx���0  �          @�
=��=q�xQ�?}p�Az�C<��=q�fff?�ffA{C<�                                    Bx����  �          @�ff�ȣ׿xQ�?�33A$��C<�\�ȣ׿c�
?�(�A.�\C<�                                    Bx���|  �          @�ff��=q�Q�?n{A��C;^���=q�@  ?z�HA��C:                                    Bx���"  T          @��ʏ\�B�\?aG�@�33C:�
�ʏ\�0��?n{A�C:B�                                    Bx����  �          @���=q�J=q?@  @׮C;)��=q�:�H?O\)@�
=C:�)                                    Bx���n  �          @����ff��z�?���A"�\C>�{��ff����?��HA.{C=�{                                    Bx���  �          @�(���
=��ff?s33A��C=����
=�z�H?��\A33C<�q                                    Bx���  �          @��
��
=���\?h��A�HC=G���
=�s33?z�HA�C<�                                    Bx���`  �          @�33�ƸR�O\)?�  Ap�C;h��ƸR�:�H?��Ap�C:�q                                    Bx��  T          @ʏ\�ƸR�#�
?���A�C9޸�ƸR�\)?�{A!�C9&f                                    Bx���  T          @��H��  �#�
?W
=@��HC9޸��  �z�?aG�@��C9L�                                    Bx��%R  "          @�=q��  �0��?.{@�C:W
��  �#�
?:�H@��
C9�H                                    Bx��3�  T          @�=q�ȣ׿!G�>��@���C9�R�ȣ׿
=?�@�ffC9c�                                    Bx��B�  �          @��H�ə��!G�>���@*�HC9�R�ə��(�>�{@Dz�C9�                                    Bx��QD  �          @�G���Q��>��@mp�C8����Q�   >�G�@���C8�=                                    Bx��_�  �          @�=q��G���>�
=@s�
C8��G����H>�@���C8z�                                    Bx��n�  �          @�33��=q���>�p�@U�C7� ��=q�Ǯ>���@eC7}q                                    Bx��}6  T          @�����Q쾣�
>�33@N{C6�H��Q쾔z�>\@[�C6��                                    Bx����  "          @�����Q��ff>�=q@{C8
��Q��(�>���@0��C7�f                                    Bx����  �          @ȣ���  ��33>�p�@VffC75���  ���
>Ǯ@dz�C6��                                    Bx���(  "          @ə���Q쾽p�?��@��RC7\)��Q쾣�
?�R@�{C6�3                                    Bx����  T          @�Q���\)���?(�@��C6h���\)�aG�?!G�@���C5�q                                    Bx���t  �          @�Q��ƸR�aG�?:�H@ָRC6\�ƸR�.{?@  @��HC5�\                                    Bx���  T          @ȣ��ƸR�aG�?Y��@��\C6��ƸR�#�
?^�R@��RC5s3                                    Bx����  �          @�  ��p����R?s33A  C6����p��u?xQ�A
=C6B�                                    Bx���f  �          @ȣ���ff���
?n{A(�C6�3��ff��  ?s33A\)C6Q�                                    Bx��  �          @Ǯ��z�Ǯ?��A��C7�)��zᾞ�R?��A ��C6޸                                    Bx���  �          @�{��=q��p�?�A-C7����=q��z�?�Q�A1p�C6��                                    Bx��X  �          @���=q��?�{A%C5=q��=q�L��?�\)A&�RC4u�                                    Bx��,�  �          @����G���?�
=A0��C5!H��G���?�
=A1p�C4L�                                    Bx��;�  �          @�����þ��?�(�A7\)C7�H���þ��
?�  A;\)C7�                                    Bx��JJ  �          @ƸR��녿�\?��RA8��C8�
��녾�
=?��
A=C7��                                    Bx��X�  �          @�{���׿!G�?��\A=G�C9�����׿�?��AC�
C9�                                    Bx��g�  T          @�p���
=�0��?�33AR=qC:�)��
=�z�?���AY��C9�
                                    Bx��v<  �          @�z�������?�p�A8��C9J=�����?��\A>�\C8ff                                    Bx����  �          @��
���þ�(�?z�HA�HC8\���þ�33?�G�A\)C7\)                                    Bx����  �          @Å���þǮ?k�A
=C7�R���þ��
?s33A33C7�                                    Bx���.  
�          @��
��  ��\?�{A'�
C8޸��  ��(�?�33A-�C8\                                    Bx����  
�          @�����z�+�?���Aq�C:���z���?�33Ax��C9W
                                    Bx���z  T          @�z���z�!G�?У�Av=qC:)��z��\?�A|��C8�f                                    Bx���   �          @��
��p��(�?�z�AUC9�H��p��   ?��HA\(�C8�
                                    Bx����  T          @����
=���?�z�AS\)C9��
=���H?���AYC8�R                                    Bx���l  "          @�������?�z�Az=qC8�������33?�Q�A
=C7h�                                    Bx���  T          @���ff�
=q?�ffAip�C90���ff��
=?˅Ao
=C8
=                                    Bx���  T          @�p����R�
=?�p�A^{C9�����R��?\AdQ�C8�
                                    Bx��^  "          @�p����R���?\Ac�C8�����R��33?�ffAhQ�C7aH                                    Bx��&  �          @ƸR�����=q?���Ak�C6������\)?˅Am�C5aH                                    Bx��4�  T          @����{��ff?ǮAj�\C8Q���{����?˅Ao33C7&f                                    Bx��CP  �          @������Ϳ&ff?�=qAn{C:B����Ϳ�?�\)Au�C9�                                    Bx��Q�  T          @��������?�  Ab�RC9����?�ffAiG�C8��                                    Bx��`�  �          @������R�
=q?�AU�C9!H���R��
=?���AZ�HC8\                                    Bx��oB  �          @�z����R�&ff?�=qAH  C::����R���?���AO
=C9:�                                    Bx��}�  T          @������׿:�H?�G�A�C:�����׿&ff?���A�
C:#�                                    Bx����  �          @�p���  �(�?��AC�C9�
��  ��\?���AJffC8ٚ                                    Bx���4  �          @�{��G��8Q�?���A((�C:����G��!G�?�
=A0(�C9�R                                    Bx����  �          @�
=���H�8Q�?�ffA(�C:� ���H�!G�?�{A$(�C9�3                                    